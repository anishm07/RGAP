"""
Generates the two figures for the per-head deactivation study:
  1. A 12x12 heatmap (layers x heads), diverging colormap, like EffiSelecViT's Table 3.
  2. A histogram of per-head delta-accuracy values, like EffiSelecViT's Fig 2(a).

Usage:
    python plot_head_deactivation.py --results-dir ./head_deactivation_results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main(args):
    results_dir = Path(args.results_dir)
    delta_matrix = np.load(results_dir / 'delta_acc_matrix.npy')
    with open(results_dir / 'summary.json') as f:
        summary = json.load(f)

    num_layers, num_heads = delta_matrix.shape

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = np.max(np.abs(delta_matrix))
    im = ax.imshow(delta_matrix, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([str(h + 1) for h in range(num_heads)])
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(l + 1) for l in range(num_layers)])
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title(
        f"Delta Top-1 Accuracy when single head is deactivated\n"
        f"(Baseline: {summary['baseline_top1_acc']:.2f}%, ImageNet-100 val)"
    )

    for l in range(num_layers):
        for h in range(num_heads):
            ax.text(h, l, f"{delta_matrix[l, h]:.2f}", ha='center', va='center', fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Delta Top-1 Accuracy (pp)')
    fig.tight_layout()
    fig.savefig(results_dir / 'head_deactivation_heatmap.png', dpi=300)
    fig.savefig(results_dir / 'head_deactivation_heatmap.pdf')
    plt.close(fig)

    # --- Histogram ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(delta_matrix.flatten(), bins=20, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=1, label='No change')
    ax.set_xlabel('Delta Top-1 Accuracy (pp)')
    ax.set_ylabel('Head count')
    ax.set_title('Distribution of per-head deactivation impact (DeiT-Base, ImageNet-100)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / 'head_deactivation_histogram.png', dpi=300)
    fig.savefig(results_dir / 'head_deactivation_histogram.pdf')
    plt.close(fig)

    # --- Per-layer mean delta (supports your "layers 6-7 protected" claim) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    layer_means = delta_matrix.mean(axis=1)
    ax.bar(range(1, num_layers + 1), layer_means)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Delta Top-1 Accuracy (pp)')
    ax.set_title('Mean per-head deactivation impact by layer')
    ax.set_xticks(range(1, num_layers + 1))
    fig.tight_layout()
    fig.savefig(results_dir / 'head_deactivation_per_layer_mean.png', dpi=300)
    fig.savefig(results_dir / 'head_deactivation_per_layer_mean.pdf')
    plt.close(fig)

    print(f"Figures saved to {results_dir}")
    print(f"Fraction of heads classified redundant (delta >= 0): "
          f"{summary['fraction_redundant']:.2%}")

    # --- Zeta vs. Delta-Accuracy scatter (key validation plot) ---
    zeta_path = results_dir / 'starting_head_zeta.npy'
    if zeta_path.exists():
        starting_zeta = np.load(zeta_path)
        corr = summary.get('zeta_vs_delta_correlation', None)
        if corr is None or np.std(starting_zeta) < 1e-8:
            print("Skipping zeta-vs-delta scatter: zeta has ~zero variance "
                  "(this is Version A, the raw pretrained checkpoint -- "
                  "expected, not an error).")
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(
                starting_zeta.flatten(), delta_matrix.flatten(),
                c=np.repeat(np.arange(num_layers), num_heads),
                cmap='viridis', alpha=0.8
            )
            ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
            ax.set_xlabel('Learned head_zeta (post-search, pre-pruning)')
            ax.set_ylabel('Delta Top-1 Accuracy when head is deactivated (pp)')
            title = f'Learned zeta vs. measured deactivation impact\n(Pearson r = {corr:.3f})'
            ax.set_title(title)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Layer index')
            fig.tight_layout()
            fig.savefig(results_dir / 'zeta_vs_delta_scatter.png', dpi=300)
            fig.savefig(results_dir / 'zeta_vs_delta_scatter.pdf')
            plt.close(fig)
            print(f"Zeta-vs-delta correlation: r = {corr:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot head deactivation results')
    parser.add_argument('--results-dir', required=True, type=str)
    args = parser.parse_args()
    main(args)
