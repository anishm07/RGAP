"""
utils.py  (extended for AdaptiveEffiSelecViT)
----------------------------------------------
All original helpers are kept intact.
New additions:
  - get_adaptive_threshold_from_model()  : layer-wise threshold from live model
  - log_redundancy_stats()               : pretty-print per-layer stats
  - save_redundancy_info()               : save stats to JSON for analysis
"""

import io
import os
import json
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Original helpers — unchanged
# -----------------------------------------------------------------------

def get_head_zeta(model):
    head_zeta = []
    for name, param in model.named_parameters():
        if 'head_zeta' in name:
            head_zeta.append(param.cpu().detach().reshape(-1).numpy().tolist())
    return [z for k in head_zeta for z in k]


def get_mlp_zeta(model):
    mlp_zeta = []
    for name, param in model.named_parameters():
        if 'mlp_zeta' in name:
            mlp_zeta.append(param.cpu().detach().reshape(-1).numpy().tolist())
    return [z for k in mlp_zeta for z in k]


def get_threshold(checkpoint_path, head_prune_ratio, mlp_prune_ratio):
    """Original global-ratio threshold function — kept for baseline runs."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    head_zeta, mlp_zeta = [], []
    for i in checkpoint['model']:
        if 'head_zeta' in i:
            head_zeta.append(checkpoint['model'][i])
        if 'mlp_zeta' in i:
            mlp_zeta.append(checkpoint['model'][i])

    head_threshold = mlp_threshold = None

    if head_zeta:
        head_data = sorted([z for k in head_zeta for z in k.squeeze().reshape(-1).numpy().tolist()])
        idx = int(head_prune_ratio * len(head_data))
        head_threshold = head_data[idx - 1]
        print(f'head_threshold={head_threshold}')

    if mlp_zeta:
        mlp_data = sorted([z for k in mlp_zeta for z in k.squeeze().reshape(-1).numpy().tolist()])
        idx = int(mlp_prune_ratio * len(mlp_data))
        mlp_threshold = mlp_data[idx - 1]
        print(f'mlp_threshold={mlp_threshold}')

    return [head_threshold, mlp_threshold]


# -----------------------------------------------------------------------
# NEW: Layer-wise adaptive threshold (uses live model + RedundancyWeights)
# -----------------------------------------------------------------------

def get_adaptive_threshold_from_model(model,
                                       redundancy_weights,
                                       head_rho_min=0.10,
                                       head_rho_max=0.60,
                                       mlp_rho_min=0.10,
                                       mlp_rho_max=0.70,
                                       device="cpu"):
    """
    Compute per-layer thresholds directly from a live model (no checkpoint
    file needed). Used during the search phase to monitor evolving ratios.

    Returns
    -------
    head_thresholds : list[float]
    mlp_thresholds  : list[float]
    redundancy_info : dict  (for logging / TensorBoard)
    """
    from adaptive_pruning import scores_to_ratios

    def _stats_and_thresholds(zeta_dict, rho_min, rho_max):
        if not zeta_dict:
            return [], [], {}

        layers = sorted(zeta_dict.keys())
        mu_l, sigma_l, H_l, raw_l = [], [], [], []

        for idx in layers:
            z = zeta_dict[idx].to(device)
            mu    = z.mean()
            sigma = z.std() if z.numel() > 1 else torch.zeros(1, device=device).squeeze()
            p     = F.softmax(z, dim=0)
            H     = -(p * torch.log(p.clamp(min=1e-8))).sum()
            mu_l.append(mu); sigma_l.append(sigma)
            H_l.append(H);   raw_l.append(z)

        mu_t    = torch.stack(mu_l)
        sigma_t = torch.stack(sigma_l)
        H_t     = torch.stack(H_l)

        with torch.no_grad():
            S   = redundancy_weights(mu_t, sigma_t, H_t)
            rho = scores_to_ratios(S, rho_min=rho_min, rho_max=rho_max)

        thresholds = []
        for z, r in zip(raw_l, rho.tolist()):
            sorted_z = z.detach().sort().values
            n_prune  = max(1, int(r * len(sorted_z)))
            thresholds.append(sorted_z[n_prune - 1].item())

        info = {
            "S":     S.detach().cpu().tolist(),
            "rho":   rho.detach().cpu().tolist(),
            "mu":    [m.item() for m in mu_l],
            "sigma": [s.item() for s in sigma_l],
            "H":     [h.item() for h in H_l],
        }
        return thresholds, rho.detach().cpu().tolist(), info

    # Collect zetas from live model
    head_z, mlp_z = {}, {}
    for name, param in model.named_parameters():
        if 'head_zeta' in name:
            idx = int(name.split('.')[1])
            head_z[idx] = param.detach().reshape(-1)
        if 'mlp_zeta' in name:
            idx = int(name.split('.')[1])
            mlp_z[idx] = param.detach().reshape(-1)

    head_thresh, head_ratios, head_info = _stats_and_thresholds(
        head_z, head_rho_min, head_rho_max)
    mlp_thresh,  mlp_ratios,  mlp_info  = _stats_and_thresholds(
        mlp_z,  mlp_rho_min,  mlp_rho_max)

    redundancy_info = {
        "head": head_info,
        "mlp":  mlp_info,
        "alpha": redundancy_weights.alpha.item(),
        "beta":  redundancy_weights.beta.item(),
        "gamma": redundancy_weights.gamma.item(),
    }
    return head_thresh, mlp_thresh, head_ratios, mlp_ratios, redundancy_info


# -----------------------------------------------------------------------
# NEW: Logging helpers
# -----------------------------------------------------------------------

def log_redundancy_stats(redundancy_info: dict, epoch: int = -1):
    """Pretty-print per-layer redundancy statistics to stdout."""
    tag = f"[Epoch {epoch}]" if epoch >= 0 else ""
    print(f"\n{'='*60}")
    print(f"  Redundancy Stats {tag}")
    print(f"  α={redundancy_info['alpha']:.4f}  "
          f"β={redundancy_info['beta']:.4f}  "
          f"γ={redundancy_info['gamma']:.4f}")
    for key in ("head", "mlp"):
        info = redundancy_info.get(key, {})
        if not info:
            continue
        print(f"\n  [{key.upper()}]")
        print(f"  {'Layer':>6}  {'S':>7}  {'ρ':>6}  {'μ':>7}  {'σ':>7}  {'H':>7}")
        for i, (s, r, m, sig, h) in enumerate(zip(
                info["S"], info["rho"], info["mu"], info["sigma"], info["H"])):
            print(f"  {i:>6}  {s:>7.4f}  {r:>6.3f}  {m:>7.4f}  {sig:>7.4f}  {h:>7.4f}")
    print('='*60)


def save_redundancy_info(redundancy_info: dict, save_path: str):
    """Save full redundancy statistics to a JSON file for offline analysis."""
    with open(save_path, 'w') as f:
        json.dump(redundancy_info, f, indent=2)
    print(f"[AdaptivePruning] Redundancy info saved to {save_path}")


# -----------------------------------------------------------------------
# Original distributed / logging utilities — unchanged below
# -----------------------------------------------------------------------

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg,
            global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{n}: {str(m)}" for n, m in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        header = header or ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}',
                   '{meters}', 'time: {time}', 'data: {data}']
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable), eta=eta_string,
                                         meters=str(self), time=str(iter_time),
                                         data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string,
                                         meters=str(self), time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')


def _load_checkpoint_for_ema(model_ema, checkpoint):
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema': checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                          init_method=args.dist_url,
                                          world_size=args.world_size,
                                          rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
