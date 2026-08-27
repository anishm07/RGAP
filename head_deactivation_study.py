"""
Per-head deactivation study (EffiSelecViT Table 3 / Fig 2(a) analogue) for RGAP.

Uses the checkpoint saved at the END of adaptive_search.py's 10-epoch search
phase (the one that achieved the reported 91.90% unpruned-reference accuracy)
-- i.e. the model BEFORE any structural pruning is applied. At this point all
heads/neurons are still structurally present, but head_zeta/mlp_zeta values
are already continuous, learned values shaped by 10 epochs of L1-regularized
training (not uniformly 1.0).

For each attention head, we additionally zero out its head_zeta (on top of
whatever value the search phase already learned for it), evaluate Top-1
accuracy on the ImageNet-100 validation set, and record the delta vs. the
un-modified (post-search, pre-pruning) baseline. This tests whether RGAP's
learned zeta ranking (and hence its mu/sigma/H statistics and resulting
ratio rho_l) lines up with each head's actual, independently-measured
contribution to accuracy.

Usage:
    python head_deactivation_study.py \
        --model deit_base_patch16_224 \
        --data-path /path/to/imagenet100 \
        --resume /path/to/adaptive_search_best_checkpoint.pth \
        --output-dir ./head_deactivation_results

Outputs:
    - delta_acc_matrix.csv   (12 layers x 12 heads, Δacc values)
    - delta_acc_matrix.npy   (same, as numpy array, for plotting)
    - summary.json           (baseline acc, #harmful heads, #redundant heads,
                               and the starting head_zeta distribution)
"""
import argparse
import json
import csv
from pathlib import Path

import numpy as np
import torch

# These imports assume this script sits alongside the project files
# (models.py, vision_transformer.py, datasets.py, engine.py).
import models  # noqa: F401  (registers deit_* models via timm registry)
from datasets import build_dataset
from engine import evaluate
from timm.models import create_model
import utils


def get_args_parser():
    parser = argparse.ArgumentParser('Per-head deactivation study', add_help=False)
    parser.add_argument('--model', default='deit_base_patch16_224', type=str)
    parser.add_argument('--data-path', required=True, type=str)
    parser.add_argument('--data-set', default='IMNET', type=str)
    parser.add_argument('--resume', required=True, type=str,
                         help='Path to the checkpoint saved by adaptive_search.py at the END '
                              'of the 10-epoch search phase (the one that achieved your '
                              'reported 91.90%% unpruned baseline) -- i.e. the model BEFORE '
                              'structural pruning is applied, with learned (non-uniform) '
                              'head_zeta / mlp_zeta values.')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true', default=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output-dir', default='./head_deactivation_results', type=str)
    parser.add_argument('--input-size', default=224, type=int)
    # Args required by datasets.py's build_transform/build_dataset, even
    # though we only call them with is_train=False:
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float)
    parser.add_argument('--inat-category', default='name', type=str)
    parser.add_argument('--color-jitter', default=0.3, type=float)
    parser.add_argument('--aa', default='rand-m9-mstd0.5-inc1', type=str)
    parser.add_argument('--train-interpolation', default='bicubic', type=str)
    parser.add_argument('--reprob', default=0.25, type=float)
    parser.add_argument('--remode', default='pixel', type=str)
    parser.add_argument('--recount', default=1, type=int)
    parser.add_argument('--num-classes', default=None, type=int,
                         help='Override num_classes. Required for ImageNet-100 since '
                              "datasets.py's build_dataset() hardcodes nb_classes=1000 "
                              "for data_set='IMNET' regardless of actual folder count.")
    return parser


@torch.no_grad()
def set_head_zeta(model, layer_idx, head_idx, value):
    """Set a single head's zeta value in-place. Returns the previous value."""
    attn = model.blocks[layer_idx].attn
    prev = attn.head_zeta.data[0, 0, head_idx, 0, 0].item()
    attn.head_zeta.data[0, 0, head_idx, 0, 0] = value
    return prev


def main(args):
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build validation set / loader (reuse the project's existing pipeline)
    dataset_val, num_classes = build_dataset(is_train=False, args=args)
    if args.num_classes is not None:
        print(f"Overriding detected num_classes={num_classes} with --num-classes={args.num_classes}")
        num_classes = args.num_classes
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Build model with head_search=True, mlp_search=True (matching
    # adaptive_search.py's model construction), finetune=False so that
    # qkv *= head_zeta is applied directly rather than via threshold.
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        head_search=True,
        mlp_search=True,
        finetune=False,
    )
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded checkpoint with message:', msg)
    model.to(device)
    model.eval()

    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads

    # NOTE: this checkpoint is the output of the 10-epoch search phase
    # (adaptive_search.py), evaluated BEFORE structural pruning is applied.
    # head_zeta values are therefore continuous, learned values (not all 1.0) —
    # the L1 penalty has already shaped them, but no components have been
    # physically removed yet, so this remains a valid "unpruned" reference
    # for FLOPs/structure purposes. We log the starting zeta distribution
    # for transparency rather than asserting it is uniform.
    print("\n=== head_zeta distribution at load time (per layer) ===")
    for l in range(num_layers):
        z = model.blocks[l].attn.head_zeta.data.flatten()
        print(f"Layer {l:2d} | min={z.min().item():.4f} max={z.max().item():.4f} "
              f"mean={z.mean().item():.4f} std={z.std().item():.4f}")

    # Baseline accuracy (all heads active, at their learned zeta values)
    print("\nEvaluating full (unpruned) model baseline...")
    baseline_stats = evaluate(data_loader_val, model, device)
    baseline_acc = baseline_stats['acc1']
    print(f"Baseline Top-1 accuracy: {baseline_acc:.4f}%")

    delta_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)
    raw_acc_matrix = np.zeros((num_layers, num_heads), dtype=np.float64)

    for l in range(num_layers):
        for h in range(num_heads):
            prev = set_head_zeta(model, l, h, 0.0)
            stats = evaluate(data_loader_val, model, device)
            acc = stats['acc1']
            delta = acc - baseline_acc
            delta_matrix[l, h] = delta
            raw_acc_matrix[l, h] = acc
            set_head_zeta(model, l, h, prev)  # restore
            print(f"Layer {l:2d} Head {h:2d} | Acc={acc:.4f}% | Delta={delta:+.4f} pp")

    # Save raw matrix
    np.save(Path(args.output_dir) / 'delta_acc_matrix.npy', delta_matrix)
    np.save(Path(args.output_dir) / 'raw_acc_matrix.npy', raw_acc_matrix)

    with open(Path(args.output_dir) / 'delta_acc_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer\\head'] + [str(h) for h in range(num_heads)])
        for l in range(num_layers):
            writer.writerow([l] + [f"{delta_matrix[l, h]:.4f}" for h in range(num_heads)])

    # Summary statistics
    harmful = int(np.sum(delta_matrix < 0))   # removing head hurts accuracy
    redundant = int(np.sum(delta_matrix >= 0))  # removing head helps or is neutral
    total = num_layers * num_heads

    # Capture the starting (learned, post-search) zeta values for correlation
    # analysis against the measured delta-accuracy values.
    starting_zeta = np.zeros((num_layers, num_heads), dtype=np.float64)
    for l in range(num_layers):
        starting_zeta[l, :] = model.blocks[l].attn.head_zeta.data.flatten().cpu().numpy()
    np.save(Path(args.output_dir) / 'starting_head_zeta.npy', starting_zeta)

    # Correlation between learned zeta and measured delta-accuracy:
    # if RGAP's zeta ranking is meaningful, heads with LOW learned zeta
    # should tend to have delta >= 0 (safe/redundant to remove), and heads
    # with HIGH learned zeta should have delta < 0 (harmful to remove).
    zeta_flat = starting_zeta.flatten()
    delta_flat = delta_matrix.flatten()
    if np.std(zeta_flat) < 1e-8:
        # Version A (raw pretrained checkpoint): zeta is uniformly ~1.0,
        # so a correlation is undefined/meaningless. This is expected and fine.
        correlation = None
        print("\nNote: head_zeta has ~zero variance (raw pretrained checkpoint, "
              "no search-phase training applied). Skipping zeta-vs-delta "
              "correlation -- this is expected for the EffiSelecViT-style "
              "Table 3 replication (Version A).")
    else:
        correlation = float(np.corrcoef(zeta_flat, delta_flat)[0, 1])

    summary = {
        'model': args.model,
        'baseline_top1_acc': baseline_acc,
        'num_layers': num_layers,
        'num_heads_per_layer': num_heads,
        'total_heads': total,
        'num_harmful_heads': harmful,        # delta < 0
        'num_redundant_heads': redundant,    # delta >= 0
        'fraction_redundant': redundant / total,
        'max_delta': float(np.max(delta_matrix)),
        'max_delta_location': [int(x) for x in np.unravel_index(np.argmax(delta_matrix), delta_matrix.shape)],
        'min_delta': float(np.min(delta_matrix)),
        'min_delta_location': [int(x) for x in np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)],
        'mean_delta_per_layer': delta_matrix.mean(axis=1).tolist(),
        'mean_zeta_per_layer': starting_zeta.mean(axis=1).tolist(),
        'zeta_vs_delta_correlation': correlation,
        'zeta_vs_delta_interpretation': (
            'Negative correlation expected if RGAP zeta ranking is meaningful: '
            'low zeta should predict high (safe) delta, high zeta should predict '
            'low (harmful) delta.'
        ),
    }
    with open(Path(args.output_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Head deactivation study', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
