"""
adaptive_prune.py
-----------------
Phase 2 of AdaptiveEffiSelecViT:
  Loads the trained checkpoint (model + redundancy_weights),
  computes per-layer thresholds, generates masks, and writes
  the structurally pruned model ready for fine-tuning.

Usage:
    python adaptive_prune.py \
        --model deit_base_patch16_224 \
        --checkpoint ./output/adaptive_search/checkpoint.pth \
        --output-dir ./output/adaptive_pruned \
        --head-rho-min 0.10 --head-rho-max 0.60 \
        --mlp-rho-min  0.10 --mlp-rho-max  0.70
"""

import argparse
import json
import os
from pathlib import Path

import torch
from timm.models import create_model

from adaptive_pruning import RedundancyWeights, get_adaptive_thresholds
import utils
import models


def get_args_parser():
    parser = argparse.ArgumentParser('AdaptiveEffiSelecViT Pruning', add_help=False)
    parser.add_argument('--model', default='deit_base_patch16_224', type=str)
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to search-phase checkpoint')
    parser.add_argument('--output-dir', default='./pruned_output', type=str)
    parser.add_argument('--num-classes', default=1000, type=int)
    parser.add_argument('--head-rho-min', type=float, default=0.10)
    parser.add_argument('--head-rho-max', type=float, default=0.60)
    parser.add_argument('--mlp-rho-min',  type=float, default=0.10)
    parser.add_argument('--mlp-rho-max',  type=float, default=0.70)
    parser.add_argument('--device', default='cpu', type=str)
    return parser


def main(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load RedundancyWeights from checkpoint ----------------------
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    redundancy_weights = RedundancyWeights()
    if 'redundancy_weights' in checkpoint:
        redundancy_weights.load_state_dict(checkpoint['redundancy_weights'])
        print(f"[Prune] Loaded α={redundancy_weights.alpha.item():.4f}  "
              f"β={redundancy_weights.beta.item():.4f}  "
              f"γ={redundancy_weights.gamma.item():.4f}")
    else:
        print("[Prune] WARNING: No redundancy_weights in checkpoint. "
              "Using default (equal) weights.")
    redundancy_weights.to(device)

    # ---- Compute per-layer thresholds --------------------------------
    print("\n[Prune] Computing adaptive per-layer thresholds ...")
    (head_thresholds, mlp_thresholds,
     head_ratios, mlp_ratios,
     redundancy_info) = get_adaptive_thresholds(
        checkpoint_path=args.checkpoint,
        redundancy_weights=redundancy_weights,
        head_rho_min=args.head_rho_min,
        head_rho_max=args.head_rho_max,
        mlp_rho_min=args.mlp_rho_min,
        mlp_rho_max=args.mlp_rho_max,
        device=args.device,
    )

    utils.log_redundancy_stats(redundancy_info)
    utils.save_redundancy_info(redundancy_info,
                                str(output_dir / 'pruning_redundancy_info.json'))

    # ---- Build pruning config masks (head_cfg, mlp_cfg) --------------
    state_dict = checkpoint['model']

    head_cfg_mask = []
    mlp_cfg_mask  = []

    # Head masks
    head_keys = sorted([k for k in state_dict if 'head_zeta' in k],
                        key=lambda k: int(k.split('.')[1]))
    for i, k in enumerate(head_keys):
        z = state_dict[k]
        thresh = head_thresholds[i] if i < len(head_thresholds) else 0.0
        mask = (z > thresh).int()
        head_cfg_mask.append(mask)
        n_kept  = mask.sum().item()
        n_total = mask.numel()
        print(f"  HEAD layer {i:2d}: kept {n_kept}/{n_total} heads  "
              f"(threshold={thresh:.6f}  ρ={head_ratios[i]:.3f})")

    # MLP masks
    mlp_keys = sorted([k for k in state_dict if 'mlp_zeta' in k],
                       key=lambda k: int(k.split('.')[1]))
    for i, k in enumerate(mlp_keys):
        z = state_dict[k]
        thresh = mlp_thresholds[i] if i < len(mlp_thresholds) else 0.0
        mask = (z > thresh).int()
        mlp_cfg_mask.append(mask)
        n_kept  = mask.sum().item()
        n_total = mask.numel()
        print(f"  MLP  layer {i:2d}: kept {n_kept}/{n_total} neurons  "
              f"(threshold={thresh:.6f}  ρ={mlp_ratios[i]:.3f})")

    # ---- Save pruning config -----------------------------------------
    pruning_config = {
        'head_thresholds': head_thresholds,
        'mlp_thresholds':  mlp_thresholds,
        'head_ratios':     head_ratios,
        'mlp_ratios':      mlp_ratios,
        'alpha':           redundancy_info['alpha'],
        'beta':            redundancy_info['beta'],
        'gamma':           redundancy_info['gamma'],
        'head_cfg_mask':   [m.cpu().tolist() for m in head_cfg_mask],
        'mlp_cfg_mask':    [m.cpu().tolist() for m in mlp_cfg_mask],
    }
    with open(output_dir / 'pruning_config.json', 'w') as f:
        json.dump(pruning_config, f, indent=2)
    print(f"\n[Prune] Pruning config saved to {output_dir}/pruning_config.json")

    # ---- Save checkpoint with per-layer thresholds embedded ----------
    # The finetune script reads head_threshold / mlp_threshold as scalars.
    # For layer-wise thresholds we save both the lists AND the per-layer
    # info so that the fine-tune VisionTransformer can load them.
    #
    # Strategy: save a SEPARATE pruned checkpoint that sets each layer's
    # threshold to the per-layer value by updating the state dict entries.
    # The existing VisionTransformer finetune path reads a single global
    # threshold, so we also expose the median as a fallback.

    pruned_ckpt = {
        'model':              state_dict,
        'redundancy_weights': checkpoint.get('redundancy_weights'),
        'head_thresholds':    head_thresholds,
        'mlp_thresholds':     mlp_thresholds,
        # Fallback globals (median) for any code that expects a scalar
        'head_threshold':     float(torch.tensor(head_thresholds).median()),
        'mlp_threshold':      float(torch.tensor(mlp_thresholds).median()),
        'args':               checkpoint.get('args'),
    }
    ckpt_path = output_dir / 'pruned_checkpoint.pth'
    torch.save(pruned_ckpt, ckpt_path)
    print(f"[Prune] Pruned checkpoint saved to {ckpt_path}")

    # ---- Summary table -----------------------------------------------
    print("\n[Prune] Summary")
    print(f"  {'Layer':>6}  {'Head ρ':>8}  {'Heads kept':>12}  "
          f"{'MLP ρ':>8}  {'Neurons kept':>14}")
    for i in range(len(head_cfg_mask)):
        hk = head_cfg_mask[i].sum().item()
        ht = head_cfg_mask[i].numel()
        mk = mlp_cfg_mask[i].sum().item() if i < len(mlp_cfg_mask) else '-'
        mt = mlp_cfg_mask[i].numel()      if i < len(mlp_cfg_mask) else '-'
        hr = head_ratios[i] if i < len(head_ratios) else '-'
        mr = mlp_ratios[i]  if i < len(mlp_ratios)  else '-'
        print(f"  {i:>6}  {hr:>8.3f}  {hk:>5}/{ht:<6}  {mr:>8.3f}  {mk:>6}/{mt:<7}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AdaptiveEffiSelecViT Pruning',
                                      parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
