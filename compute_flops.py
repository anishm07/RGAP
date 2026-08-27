import torch
import json
import sys
sys.path.insert(0, '/workspace/project')

with open('/workspace/output/pruned/pruning_config.json') as f:
    cfg = json.load(f)

L    = 12
H    = 12
D    = 768
D_mlp = 3072
N    = 197
d_h  = D // H

# ── Helper: flatten any nested list to 1D ────────────────────────────
def flatten(x):
    if isinstance(x, (int, float)):
        return [x]
    result = []
    for item in x:
        result.extend(flatten(item))
    return result

# ── Original FLOPs ────────────────────────────────────────────────────
flops_attn_orig = 3*N*D*D + 2*H*(N**2)*d_h + N*H*d_h*H*d_h
flops_mlp_orig  = N * 2 * D * D_mlp
orig_G = L * (flops_attn_orig + flops_mlp_orig) / 1e9
print(f"Original DeiT-B FLOPs: {orig_G:.4f} GFLOPs")

# ── Per-layer pruned FLOPs ────────────────────────────────────────────
print(f"\n  {'Layer':>5}  {'Heads':>7}  {'Neurons':>9}  {'ATTN G':>8}  {'MLP G':>7}  {'Total G':>8}")
print("  " + "-"*55)

total_pruned = 0
for i in range(L):
    hm = flatten(cfg['head_cfg_mask'][i])
    mm = flatten(cfg['mlp_cfg_mask'][i])

    h_kept = int(sum(hm))
    n_kept = int(sum(mm))

    D_pruned   = h_kept * d_h
    f_attn = 3*N*D*D_pruned + 2*h_kept*(N**2)*d_h + N*D_pruned*D
    f_mlp  = N * 2 * D * n_kept
    f_tot  = f_attn + f_mlp
    total_pruned += f_tot

    print(f"  {i:>5}  {h_kept:>4}/12  {n_kept:>6}/3072"
          f"  {f_attn/1e9:>8.4f}  {f_mlp/1e9:>7.4f}  {f_tot/1e9:>8.4f}")

pruned_G  = total_pruned / 1e9
retention = (pruned_G / orig_G) * 100
reduction = 100 - retention

print("\n" + "="*52)
print("  FLOPS COMPARISON SUMMARY")
print("="*52)
print(f"  Original DeiT-B:        {orig_G:.4f} GFLOPs")
print(f"  AdaptiveEffiSelecViT:   {pruned_G:.4f} GFLOPs")
print(f"  FLOPs retained:         {retention:.2f}%")
print(f"  FLOPs reduced:          {reduction:.2f}%")
print(f"  Accuracy (unpruned):    91.90%")
print(f"  Accuracy (pruned):      93.34%")
print(f"  Accuracy change:        +1.44%")
print("="*52)
print(f"\n  EffiSelecViT-B retains: 64.00%")
print(f"  Your method retains:    {retention:.2f}%")
