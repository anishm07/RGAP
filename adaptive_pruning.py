"""
adaptive_pruning.py
--------------------
Implements the Adaptive Redundancy-Guided Pruning Ratio module.

For each layer ℓ the redundancy score is:
    S_ℓ = α(1 − μ_ℓ) + β·σ_ℓ + γ·H_ℓ

where
    μ_ℓ  = mean of learned zeta scores       (importance)
    σ_ℓ  = std  of learned zeta scores       (diversity)
    H_ℓ  = entropy of softmax(zeta scores)   (uncertainty)
    α, β, γ are learnable scalars, jointly optimised with the zetas.

The per-layer pruning ratio is then:
    ρ_ℓ = ρ_min + (ρ_max − ρ_min) · (S_ℓ − S_min) / (S_max − S_min + ε)

This replaces the single global head_prune_ratio / mlp_prune_ratio used
in the original EffiSelecViT get_threshold() call.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Learnable weighting of the three redundancy signals
# ---------------------------------------------------------------------------

class RedundancyWeights(nn.Module):
    """
    Three learnable scalars α, β, γ that weight the three components of S_ℓ.
    Initialised to 1/3 each so that all signals start with equal influence.
    Constrained to be non-negative via softplus so the score stays meaningful.

    For ablation studies (A1), pass freeze_signals to lock specific signals
    to zero contribution and disable their gradients, isolating the effect
    of a single statistic (μ, σ, or H) on the redundancy score.

    Example:
        RedundancyWeights(freeze_signals=['beta', 'gamma'])  # μ-only (α active)
        RedundancyWeights(freeze_signals=['alpha', 'gamma']) # σ-only (β active)
        RedundancyWeights(freeze_signals=['alpha', 'beta'])  # H-only (γ active)
        RedundancyWeights()                                  # full S_ℓ (default)
    """

    def __init__(self, freeze_signals=None):
        super().__init__()
        freeze_signals = freeze_signals or []
        assert all(s in ('alpha', 'beta', 'gamma') for s in freeze_signals), \
            "freeze_signals must be a subset of ['alpha', 'beta', 'gamma']"

        self.frozen = set(freeze_signals)

        # Raw (unconstrained) parameters; we pass them through softplus.
        # Frozen signals are initialised to a large negative value so that
        # softplus(-12) ≈ 0, effectively zeroing their contribution, and
        # requires_grad=False keeps them at exactly that value forever.
        init_alpha = -12.0 if 'alpha' in self.frozen else 1.0
        init_beta  = -12.0 if 'beta'  in self.frozen else 1.0
        init_gamma = -12.0 if 'gamma' in self.frozen else 1.0

        self._alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=('alpha' not in self.frozen))
        self._beta  = nn.Parameter(torch.tensor(init_beta),  requires_grad=('beta'  not in self.frozen))
        self._gamma = nn.Parameter(torch.tensor(init_gamma), requires_grad=('gamma' not in self.frozen))

    @property
    def alpha(self):
        return F.softplus(self._alpha)

    @property
    def beta(self):
        return F.softplus(self._beta)

    @property
    def gamma(self):
        return F.softplus(self._gamma)

    def forward(self, mu, sigma, H):
        """
        Args:
            mu    : Tensor [L]  – per-layer mean importance

            sigma : Tensor [L]  – per-layer std of zeta scores
            H     : Tensor [L]  – per-layer entropy of zeta distribution
        Returns:
            S     : Tensor [L]  – per-layer redundancy scores
        """
        S = self.alpha * (1.0 - mu) + self.beta * sigma + self.gamma * H
        return S


# ---------------------------------------------------------------------------
# Redundancy statistics extraction from model zeta parameters
# ---------------------------------------------------------------------------

def compute_layer_redundancy(model, zeta_type: str = "head"):
    """
    Extract per-layer redundancy statistics (μ_ℓ, σ_ℓ, H_ℓ) from the
    learned zeta parameters.

    Args:
        model      : the VisionTransformer with head_zeta / mlp_zeta params
        zeta_type  : "head" or "mlp"

    Returns:
        mu_list    : list of scalar tensors, one per layer
        sigma_list : list of scalar tensors, one per layer
        H_list     : list of scalar tensors, one per layer
        raw_zetas  : list of 1-D tensors of raw zeta values per layer
    """
    mu_list    = []
    sigma_list = []
    H_list     = []
    raw_zetas  = []

    target_key = f"{zeta_type}_zeta"

    for name, param in model.named_parameters():
        if target_key not in name:
            continue

        z = param.detach().reshape(-1)        # flatten to 1-D
        raw_zetas.append(z)

        # Mean importance
        mu = z.mean()

        # Diversity (std)
        sigma = z.std() if z.numel() > 1 else torch.zeros(1, device=z.device).squeeze()

        # Entropy of the softmax distribution over scores
        p = F.softmax(z, dim=0)               # normalise to probability simplex
        # clamp to avoid log(0)
        H = -(p * torch.log(p.clamp(min=1e-8))).sum()

        mu_list.append(mu)
        sigma_list.append(sigma)
        H_list.append(H)

    return mu_list, sigma_list, H_list, raw_zetas


# ---------------------------------------------------------------------------
# Ratio mapper: S_ℓ  →  ρ_ℓ
# ---------------------------------------------------------------------------

def scores_to_ratios(S: torch.Tensor,
                     rho_min: float = 0.10,
                     rho_max: float = 0.60) -> torch.Tensor:
    """
    Budget-constrained min-max normalisation of S_ℓ into pruning ratios.

    ρ_ℓ = ρ_min + (ρ_max − ρ_min) · (S_ℓ − S_min) / (S_max − S_min + ε)

    Layers with the highest redundancy score get the highest pruning ratio.

    Args:
        S       : Tensor [L] – per-layer redundancy scores
        rho_min : minimum pruning ratio (protects every layer from over-pruning)
        rho_max : maximum pruning ratio

    Returns:
        rho : Tensor [L] – per-layer pruning ratios in [rho_min, rho_max]
    """
    eps = 1e-8
    S_min = S.min()
    S_max = S.max()
    rho = rho_min + (rho_max - rho_min) * (S - S_min) / (S_max - S_min + eps)
    return rho.clamp(rho_min, rho_max)


# ---------------------------------------------------------------------------
# Threshold computation (layer-wise, replaces get_threshold in utils.py)
# ---------------------------------------------------------------------------

def get_adaptive_thresholds(checkpoint_path: str,
                             redundancy_weights: RedundancyWeights,
                             head_rho_min: float = 0.10,
                             head_rho_max: float = 0.60,
                             mlp_rho_min:  float = 0.10,
                             mlp_rho_max:  float = 0.70,
                             device: str = "cpu"):
    """
    Compute per-layer thresholds for head and MLP pruning using the learned
    redundancy scores and the adaptive ratio mapper.

    Returns
    -------
    head_thresholds : list[float]  – one threshold per layer (head)
    mlp_thresholds  : list[float]  – one threshold per layer (mlp)
    head_ratios     : list[float]  – actual pruning ratio used per layer
    mlp_ratios      : list[float]  – actual pruning ratio used per layer
    redundancy_info : dict         – full statistics for logging / ablation
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]

    # ---- Collect raw zeta tensors per layer ---------------------------
    head_zetas_by_layer = {}
    mlp_zetas_by_layer  = {}

    for k, v in state_dict.items():
        if "head_zeta" in k:
            # key looks like "blocks.N.attn.head_zeta"
            layer_idx = int(k.split(".")[1])
            head_zetas_by_layer[layer_idx] = v.detach().reshape(-1)
        if "mlp_zeta" in k:
            layer_idx = int(k.split(".")[1])
            mlp_zetas_by_layer[layer_idx] = v.detach().reshape(-1)

    def _layer_stats(zeta_dict):
        """Return ordered lists of (mu, sigma, H, raw_zeta) across layers."""
        mu_list, sigma_list, H_list, raw_list = [], [], [], []
        for idx in sorted(zeta_dict.keys()):
            z = zeta_dict[idx].to(device)
            mu    = z.mean()
            sigma = z.std() if z.numel() > 1 else torch.zeros(1).to(device).squeeze()
            p     = F.softmax(z, dim=0)
            H     = -(p * torch.log(p.clamp(min=1e-8))).sum()
            mu_list.append(mu)
            sigma_list.append(sigma)
            H_list.append(H)
            raw_list.append(z)
        return mu_list, sigma_list, H_list, raw_list

    def _compute_thresholds(zeta_dict, rho_min, rho_max, label):
        if not zeta_dict:
            print(f"[AdaptivePruning] No {label} zetas found in checkpoint.")
            return [], [], []

        mu_l, sigma_l, H_l, raw_l = _layer_stats(zeta_dict)

        mu_t     = torch.stack(mu_l)
        sigma_t  = torch.stack(sigma_l)
        H_t      = torch.stack(H_l)

        # Redundancy scores S_ℓ (using learned α, β, γ)
        with torch.no_grad():
            S = redundancy_weights(mu_t, sigma_t, H_t)

        rho = scores_to_ratios(S, rho_min=rho_min, rho_max=rho_max)

        thresholds = []
        ratios     = []
        for i, (z, r) in enumerate(zip(raw_l, rho.tolist())):
            sorted_z = z.sort().values
            n_prune  = max(1, int(r * len(sorted_z)))
            thresh   = sorted_z[n_prune - 1].item()
            thresholds.append(thresh)
            ratios.append(r)
            print(f"  [{label}] Layer {i:2d} | S={S[i]:.4f}  ρ={r:.3f}  "
                  f"μ={mu_l[i].item():.4f}  σ={sigma_l[i].item():.4f}  "
                  f"H={H_l[i].item():.4f}  threshold={thresh:.6f}")

        return thresholds, ratios, {
            "S": S.cpu().tolist(),
            "rho": rho.cpu().tolist(),
            "mu":  [m.item() for m in mu_l],
            "sigma": [s.item() for s in sigma_l],
            "H":   [h.item() for h in H_l],
        }

    print("\n[AdaptivePruning] Computing HEAD thresholds ...")
    head_thresholds, head_ratios, head_info = _compute_thresholds(
        head_zetas_by_layer, head_rho_min, head_rho_max, "HEAD")

    print("\n[AdaptivePruning] Computing MLP thresholds ...")
    mlp_thresholds, mlp_ratios, mlp_info = _compute_thresholds(
        mlp_zetas_by_layer, mlp_rho_min, mlp_rho_max, "MLP")

    redundancy_info = {"head": head_info, "mlp": mlp_info,
                       "alpha": redundancy_weights.alpha.item(),
                       "beta":  redundancy_weights.beta.item(),
                       "gamma": redundancy_weights.gamma.item()}

    return head_thresholds, mlp_thresholds, head_ratios, mlp_ratios, redundancy_info


# ---------------------------------------------------------------------------
# Loss term: regularise α, β, γ so they don't collapse to zero
# ---------------------------------------------------------------------------

def redundancy_weight_reg_loss(redundancy_weights: RedundancyWeights,
                                lambda_reg: float = 1e-3) -> torch.Tensor:
    """
    Optional: encourage α+β+γ to stay near 1 (unit-sum constraint softened
    into a loss term).  Prevents all three collapsing to zero under L2 decay.
    """
    total = redundancy_weights.alpha + redundancy_weights.beta + redundancy_weights.gamma
    return lambda_reg * (total - 1.0) ** 2
