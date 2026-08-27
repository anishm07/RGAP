import json, sys

variant_path = sys.argv[1]

with open(variant_path) as f:
    cfg = json.load(f)

def flatten(x):
    if isinstance(x, (int, float)):
        return [x]
    result = []
    for item in x:
        result.extend(flatten(item))
    return result

L, H, D, D_mlp, N = 12, 12, 768, 3072, 197
d_h = D // H

flops_attn_orig = 3*N*D*D + 2*H*(N**2)*d_h + N*H*d_h*H*d_h
flops_mlp_orig  = N * 2 * D * D_mlp
orig_G = L * (flops_attn_orig + flops_mlp_orig) / 1e9

total_pruned = 0
for i in range(L):
    hm = flatten(cfg['head_cfg_mask'][i])
    mm = flatten(cfg['mlp_cfg_mask'][i])
    h_kept = int(sum(hm))
    n_kept = int(sum(mm))
    D_pruned = h_kept * d_h
    f_attn = 3*N*D*D_pruned + 2*h_kept*(N**2)*d_h + N*D_pruned*D
    f_mlp  = N * 2 * D * n_kept
    total_pruned += f_attn + f_mlp

pruned_G  = total_pruned / 1e9
retention = (pruned_G / orig_G) * 100
print(f"FLOPs retained: {retention:.2f}%  ({pruned_G:.4f} / {orig_G:.4f} GFLOPs)")
