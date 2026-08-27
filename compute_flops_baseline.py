import torch
import sys
sys.path.insert(0, '/workspace/project')
import utils

# ── Get the global thresholds used by baseline ────────────────────────
head_threshold, mlp_threshold = utils.get_threshold(
    checkpoint_path='/workspace/output/baseline_search_v2/checkpoint.pth',
    head_prune_ratio=0.3,
    mlp_prune_ratio=0.5
)

print(f"\nHead threshold: {head_threshold}")
print(f"MLP threshold:  {mlp_threshold}")

# ── Load the search checkpoint to get raw zetas per layer ─────────────
ckpt = torch.load('/workspace/output/baseline_search_v2/checkpoint.pth',
                   map_location='cpu', weights_only=False)
state = ckpt['model']

L, H, D, D_mlp, N = 12, 12, 768, 3072, 197
d_h = D // H

flops_attn_orig = 3*N*D*D + 2*H*(N**2)*d_h + N*H*d_h*H*d_h
flops_mlp_orig  = N * 2 * D * D_mlp
orig_G = L * (flops_attn_orig + flops_mlp_orig) / 1e9
print(f"\nOriginal DeiT-B FLOPs: {orig_G:.4f} GFLOPs")

print(f"\n  {'Layer':>5}  {'Heads':>7}  {'Neurons':>9}  {'ATTN G':>8}  {'MLP G':>7}  {'Total G':>8}")
print("  " + "-"*55)

total_pruned = 0
for i in range(L):
    head_key = f'blocks.{i}.attn.head_zeta'
    mlp_key  = f'blocks.{i}.mlp.mlp_zeta'

    h_zeta = state[head_key].reshape(-1)
    m_zeta = state[mlp_key].reshape(-1)

    h_kept = int((h_zeta > head_threshold).sum().item())
    n_kept = int((m_zeta > mlp_threshold).sum().item())

    D_pruned = h_kept * d_h
    f_attn = 3*N*D*D_pruned + 2*h_kept*(N**2)*d_h + N*D_pruned*D
    f_mlp  = N * 2 * D * n_kept
    f_tot  = f_attn + f_mlp
    total_pruned += f_tot

    print(f"  {i:>5}  {h_kept:>4}/12  {n_kept:>6}/3072"
          f"  {f_attn/1e9:>8.4f}  {f_mlp/1e9:>7.4f}  {f_tot/1e9:>8.4f}")

pruned_G  = total_pruned / 1e9
retention = (pruned_G / orig_G) * 100
reduction = 100 - retention

print("\n" + "="*55)
print("  BASELINE FLOPS SUMMARY")
print("="*55)
print(f"  Original DeiT-B:         {orig_G:.4f} GFLOPs")
print(f"  Baseline EffiSelecViT:   {pruned_G:.4f} GFLOPs")
print(f"  FLOPs retained:          {retention:.2f}%")
print(f"  FLOPs reduced:           {reduction:.2f}%")
print(f"  Accuracy (unpruned):     91.90%")
print(f"  Accuracy (pruned):       85.76%")
print(f"  Accuracy change:         -6.14%")
print("="*55)
