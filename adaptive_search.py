"""
adaptive_search.py
------------------
Phase 1 of AdaptiveEffiSelecViT:
  - Loads pre-trained ViT
  - Jointly learns zeta scores (head + MLP) AND redundancy weights (α, β, γ)
  - Uses AdaptiveSearchingDistillationLoss
  - Logs per-layer redundancy statistics each epoch
  - Saves checkpoint with both model state and redundancy_weights state

Usage (DeiT-B example):
    python adaptive_search.py \
        --model deit_base_patch16_224 \
        --pretrained-path /path/to/deit_base.pth \
        --data-path /path/to/imagenet \
        --data-set IMNET \
        --output-dir ./output/adaptive_search \
        --epochs 10 \
        --batch-size 128 \
        --lr 5e-4 \
        --head-w 1e-2 \
        --mlp-w 1e-4 \
        --lambda-reg 1e-3

All DeiT / Swin training protocol args from original search.py are supported.
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, AdaptiveSearchingDistillationLoss
from samplers import RASampler
from adaptive_pruning import RedundancyWeights
import utils
import models


# ---------------------------------------------------------------------------
# Custom training step that handles the (loss, info_dict) return value
# ---------------------------------------------------------------------------

def train_one_epoch_adaptive(model, criterion, data_loader, optimizer,
                              device, epoch, loss_scaler, max_norm=0,
                              model_ema=None, mixup_fn=None,
                              set_training_mode=True, args=None):
    """Like engine.train_one_epoch but unwraps the info dict from loss."""
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss, loss_info = criterion(samples, outputs, targets, model)

        loss_value = loss.item()

        if not torch.isfinite(loss).item():
            print(f"Loss is {loss_value}, stopping training")
            import sys; sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(**{k: v for k, v in loss_info.items()
                                 if isinstance(v, (float, int))})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Argument parser (inherits all original args + new adaptive args)
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('AdaptiveEffiSelecViT Search', add_help=False)

    # ---- Training basics -------------------------------------------------
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # ---- Model -----------------------------------------------------------
    parser.add_argument('--model', default='deit_base_patch16_224', type=str)
    parser.add_argument('--pretrained-path', default='', type=str,
                        help='Path to pre-trained ViT checkpoint')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop-path', type=float, default=0.1)
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996)
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)

    # ---- Optimiser -------------------------------------------------------
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt-eps', default=1e-8, type=float)
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip-grad', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.05)

    # ---- LR schedule -----------------------------------------------------
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None)
    parser.add_argument('--lr-noise-pct', type=float, default=0.67)
    parser.add_argument('--lr-noise-std', type=float, default=1.0)
    parser.add_argument('--warmup-lr', type=float, default=1e-6)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--decay-epochs', type=float, default=30)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--cooldown-epochs', type=int, default=10)
    parser.add_argument('--patience-epochs', type=int, default=10)
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1)

    # ---- Augmentation ----------------------------------------------------
    parser.add_argument('--color-jitter', type=float, default=0.3)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--train-interpolation', type=str, default='bicubic')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    # ---- Mixup -----------------------------------------------------------
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup-prob', type=float, default=1.0)
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5)
    parser.add_argument('--mixup-mode', type=str, default='batch')

    # ---- Distillation ----------------------------------------------------
    parser.add_argument('--teacher-model', default='regnety_160', type=str)
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none',
                        choices=['none', 'soft', 'hard'])
    parser.add_argument('--distillation-alpha', default=0.5, type=float)
    parser.add_argument('--distillation-tau', default=1.0, type=float)

    # ---- Dataset ---------------------------------------------------------
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str)
    parser.add_argument('--data-set', default='IMNET',
                        choices=['CIFAR10', 'CIFAR100', 'IMNET', 'IMNET100', 'INAT', 'INAT19'])
    parser.add_argument('--inat-category', default='name', type=str)
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float)

    # ---- Misc ------------------------------------------------------------
    parser.add_argument('--output-dir', default='', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num-workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False)
    parser.add_argument('--dist-url', default='env://')

    # ---- NEW: Adaptive pruning parameters --------------------------------
    parser.add_argument('--head-w', type=float, default=1e-2,
                        help='L1 penalty weight for head zetas (λ_head)')
    parser.add_argument('--mlp-w', type=float, default=1e-4,
                        help='L1 penalty weight for MLP zetas (λ_mlp)')
    parser.add_argument('--lambda-reg', type=float, default=1e-3,
                        help='Weight for unit-sum regularisation on α,β,γ')
    parser.add_argument('--head-rho-min', type=float, default=0.10,
                        help='Minimum per-layer head pruning ratio')
    parser.add_argument('--head-rho-max', type=float, default=0.60,
                        help='Maximum per-layer head pruning ratio')
    parser.add_argument('--mlp-rho-min', type=float, default=0.10,
                        help='Minimum per-layer MLP pruning ratio')
    parser.add_argument('--mlp-rho-max', type=float, default=0.70,
                        help='Maximum per-layer MLP pruning ratio')
    parser.add_argument('--redundancy-weights-lr', type=float, default=1e-3,
                        help='Separate LR for α,β,γ parameters')
    parser.add_argument('--freeze-signals', type=str, nargs='*', default=[],
                        choices=['alpha', 'beta', 'gamma'],
                        help='Ablation: freeze listed signals to zero contribution. '
                             'E.g. --freeze-signals beta gamma isolates mu-only (alpha active). '
                             'Leave empty for the full S_l formula (default).')

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # ---- Data -------------------------------------------------------
    dataset_train, nb_classes = build_dataset(is_train=True,  args=args)
    dataset_val,   _          = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = (RASampler(dataset_train, num_replicas=num_tasks,
                                    rank=global_rank, shuffle=True)
                         if args.repeated_aug else
                         torch.utils.data.DistributedSampler(
                             dataset_train, num_replicas=num_tasks,
                             rank=global_rank, shuffle=True))
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False)

    # ---- Mixup -------------------------------------------------------
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                         switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                         label_smoothing=args.smoothing, num_classes=nb_classes)

    # ---- Model (with head_search + mlp_search enabled) ---------------
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        head_search=True,
        mlp_search=True,
    )

    # Load pre-trained weights
    if args.pretrained_path:
        print(f"Loading pretrained weights from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        state = checkpoint.get('model', checkpoint)
        state.pop('head.weight', None)
        state.pop('head.bias', None)
        msg = model.load_state_dict(state, strict=False)
        print(f"  Missing keys: {msg.missing_keys}")
        print(f"  Unexpected keys: {msg.unexpected_keys}")
       
    model.to(device)

    # ---- RedundancyWeights (α, β, γ) ---------------------------------
    redundancy_weights = RedundancyWeights(freeze_signals=args.freeze_signals).to(device)
    if args.freeze_signals:
        print(f"[Ablation] Frozen signals: {args.freeze_signals} "
              f"(active signal(s): {[s for s in ['alpha','beta','gamma'] if s not in args.freeze_signals]})")
    print(f"RedundancyWeights initialised: "
          f"α={redundancy_weights.alpha.item():.4f}  "
          f"β={redundancy_weights.beta.item():.4f}  "
          f"γ={redundancy_weights.gamma.item():.4f}")

    # ---- EMA ---------------------------------------------------------
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay,
                             device='cpu' if args.model_ema_force_cpu else '',
                             resume='')

    # ---- Optimiser: two parameter groups
    #      (zetas + backbone at args.lr; α,β,γ at redundancy_weights_lr)
    zeta_and_backbone_params = [p for n, p in model.named_parameters()
                                 if p.requires_grad]
    rw_params = list(redundancy_weights.parameters())

    optimizer = torch.optim.AdamW([
        {'params': zeta_and_backbone_params, 'lr': args.lr,
         'weight_decay': args.weight_decay},
        {'params': rw_params, 'lr': args.redundancy_weights_lr,
         'weight_decay': 0.0},          # no decay on α,β,γ
    ])

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # ---- Loss criterion ----------------------------------------------
    if mixup_active:
        base_criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        base_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        base_criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'Need a teacher model path for distillation'
        print(f"Loading teacher model: {args.teacher_model}")
        teacher_model = create_model(args.teacher_model, pretrained=False,
                                      num_classes=nb_classes, global_pool='avg')
        checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    distil_criterion = DistillationLoss(
        base_criterion, teacher_model,
        args.distillation_type, args.distillation_alpha, args.distillation_tau)

    criterion = AdaptiveSearchingDistillationLoss(
        base_criterion=distil_criterion,
        redundancy_weights=redundancy_weights,
        device=device,
        head_w=args.head_w,
        mlp_w=args.mlp_w,
        lambda_reg=args.lambda_reg,
    )

    # ---- Output dir + TensorBoard ------------------------------------
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if utils.is_main_process() and args.output_dir:
        writer = SummaryWriter(log_dir=str(output_dir / 'tb_logs'))

    # ---- Resume ------------------------------------------------------
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'redundancy_weights' in checkpoint:
            redundancy_weights.load_state_dict(checkpoint['redundancy_weights'])
        if not args.eval and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint.get('epoch', '?')}")

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy: {test_stats['acc1']:.1f}%")
        return

    # ---- Training loop -----------------------------------------------
    print(f"\nStart adaptive search training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_adaptive(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler,
            max_norm=args.clip_grad,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            args=args,
        )

        lr_scheduler.step(epoch)

        # Log per-layer redundancy stats
        _, _, head_ratios, mlp_ratios, redundancy_info = \
            utils.get_adaptive_threshold_from_model(
                model, redundancy_weights,
                head_rho_min=args.head_rho_min,
                head_rho_max=args.head_rho_max,
                mlp_rho_min=args.mlp_rho_min,
                mlp_rho_max=args.mlp_rho_max,
                device=device)

        if utils.is_main_process():
            utils.log_redundancy_stats(redundancy_info, epoch=epoch)

            if writer:
                for k, v in train_stats.items():
                    writer.add_scalar(f'train/{k}', v, epoch)
                writer.add_scalar('redundancy/alpha', redundancy_info['alpha'], epoch)
                writer.add_scalar('redundancy/beta',  redundancy_info['beta'],  epoch)
                writer.add_scalar('redundancy/gamma', redundancy_info['gamma'], epoch)
                for i, (rh, rm) in enumerate(zip(head_ratios, mlp_ratios)):
                    writer.add_scalar(f'redundancy/head_ratio_layer{i}', rh, epoch)
                    writer.add_scalar(f'redundancy/mlp_ratio_layer{i}',  rm, epoch)

        # Evaluate
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy on val set: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats['acc1'])

        if writer and utils.is_main_process():
            writer.add_scalar('val/acc1', test_stats['acc1'], epoch)

        # Save checkpoint
        if args.output_dir:
            checkpoint_path = output_dir / 'checkpoint.pth'
            utils.save_on_master({
                'model':              model.state_dict(),
                'redundancy_weights': redundancy_weights.state_dict(),
                'optimizer':          optimizer.state_dict(),
                'lr_scheduler':       lr_scheduler.state_dict(),
                'epoch':              epoch,
                'args':               args,
            }, checkpoint_path)

            # Also save redundancy info JSON
            if utils.is_main_process():
                utils.save_redundancy_info(
                    redundancy_info,
                    str(output_dir / f'redundancy_epoch{epoch:02d}.json'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}':  v for k, v in test_stats.items()},
                     'epoch': epoch, 'max_accuracy': max_accuracy,
                     'alpha': redundancy_info['alpha'],
                     'beta':  redundancy_info['beta'],
                     'gamma': redundancy_info['gamma']}

        if args.output_dir and utils.is_main_process():
            with (output_dir / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time: {total_time_str}')
    print(f'Max accuracy:  {max_accuracy:.2f}%')

    if writer:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AdaptiveEffiSelecViT Search',
                                      parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
