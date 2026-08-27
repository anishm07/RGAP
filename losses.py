"""
losses.py  (extended for AdaptiveEffiSelecViT)
----------------------------------------------
Adds AdaptiveSearchingDistillationLoss which jointly optimises:
  1. Base cross-entropy (or distillation) loss
  2. L1 regularisation on head_zeta scores   (sparsity)
  3. L1 regularisation on mlp_zeta scores    (sparsity)
  4. Soft unit-sum regularisation on α, β, γ (prevents collapse)

The RedundancyWeights (α, β, γ) are passed in from the search script so
their gradients flow through this loss back to the weight scalars.
"""

import torch
from torch.nn import functional as F


# -----------------------------------------------------------------------
# Original losses — kept unchanged for compatibility
# -----------------------------------------------------------------------

class DistillationLoss(torch.nn.Module):
    """
    Wraps a standard criterion and adds knowledge distillation loss.
    Unchanged from original EffiSelecViT.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("Distillation enabled but model didn't return dist output.")

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        return base_loss * (1 - self.alpha) + distillation_loss * self.alpha


class SearchingDistillationLoss(torch.nn.Module):
    """Original fixed-weight search loss — kept for baseline comparison."""

    def __init__(self, base_criterion, device, head_w=1e-2, mlp_w=1e-4):
        super().__init__()
        self.base_criterion = base_criterion
        self.head_w = head_w
        self.mlp_w = mlp_w
        self.device = device

    def forward(self, inputs, outputs, labels, model):
        base_loss = self.base_criterion(inputs, outputs, labels)
        add_head_loss = torch.FloatTensor([]).to(self.device)
        add_mlp_loss  = torch.FloatTensor([]).to(self.device)

        for name, param in model.named_parameters():
            if param.requires_grad and 'head_zeta' in name:
                add_head_loss = torch.cat([add_head_loss, torch.abs(param.view(-1))])
            if param.requires_grad and 'mlp_zeta' in name:
                add_mlp_loss = torch.cat([add_mlp_loss, torch.abs(param.view(-1))])

        total_head_zeta = torch.sum(add_head_loss).to(self.device)
        total_mlp_zeta  = torch.sum(add_mlp_loss).to(self.device)

        return base_loss + self.head_w * total_head_zeta + self.mlp_w * total_mlp_zeta


# -----------------------------------------------------------------------
# NEW: Adaptive search loss that also trains α, β, γ
# -----------------------------------------------------------------------

class AdaptiveSearchingDistillationLoss(torch.nn.Module):
    """
    Search-phase loss for AdaptiveEffiSelecViT.

    L_total = L_base
            + λ_head · Σ|head_zeta|          (L1 sparsity on heads)
            + λ_mlp  · Σ|mlp_zeta|           (L1 sparsity on MLP neurons)
            + λ_reg  · (α + β + γ − 1)²      (unit-sum soft constraint)

    The unit-sum term prevents the learned redundancy weights from all
    collapsing to zero under weight decay, while allowing their relative
    magnitudes to be freely optimised by gradient descent.

    Parameters
    ----------
    base_criterion    : DistillationLoss (wraps cross-entropy + distillation)
    redundancy_weights: RedundancyWeights module holding α, β, γ
    device            : torch.device
    head_w            : λ_head (default: 1e-2, same as original paper)
    mlp_w             : λ_mlp  (default: 1e-4, same as original paper)
    lambda_reg        : weight for the unit-sum regularisation (default: 1e-3)
    """

    def __init__(self, base_criterion, redundancy_weights, device,
                 head_w=1e-2, mlp_w=1e-4, lambda_reg=1e-3):
        super().__init__()
        self.base_criterion     = base_criterion
        self.redundancy_weights = redundancy_weights
        self.head_w      = head_w
        self.mlp_w       = mlp_w
        self.lambda_reg  = lambda_reg
        self.device      = device

    def forward(self, inputs, outputs, labels, model):
        # 1. Base task loss (cross-entropy ± distillation)
        base_loss = self.base_criterion(inputs, outputs, labels)

        # 2. L1 sparsity on head zetas
        head_zetas = []
        mlp_zetas  = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'head_zeta' in name:
                head_zetas.append(torch.abs(param.view(-1)))
            if param.requires_grad and 'mlp_zeta' in name:
                mlp_zetas.append(torch.abs(param.view(-1)))

        l1_head = torch.sum(torch.cat(head_zetas)) if head_zetas else torch.tensor(0.0, device=self.device)
        l1_mlp  = torch.sum(torch.cat(mlp_zetas))  if mlp_zetas  else torch.tensor(0.0, device=self.device)

        # 3. Soft unit-sum regularisation on α, β, γ
        alpha = self.redundancy_weights.alpha
        beta  = self.redundancy_weights.beta
        gamma = self.redundancy_weights.gamma
        reg_loss = self.lambda_reg * (alpha + beta + gamma - 1.0) ** 2

        total_loss = base_loss + self.head_w * l1_head + self.mlp_w * l1_mlp + reg_loss

        return total_loss, {
            "base_loss": base_loss.item(),
            "l1_head":   l1_head.item(),
            "l1_mlp":    l1_mlp.item(),
            "reg_loss":  reg_loss.item(),
            "alpha":     alpha.item(),
            "beta":      beta.item(),
            "gamma":     gamma.item(),
        }
