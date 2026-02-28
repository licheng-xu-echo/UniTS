from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, StepLR

from .scheduler import NoamLR
from .utils import EMA


@dataclass
class ParallelModelBundle:
    model_dp: torch.nn.Module
    model_ema: torch.nn.Module
    model_ema_dp: torch.nn.Module
    ema: Optional[EMA]


def build_scheduler_from_args(args, optim):
    scheduler_type = str(getattr(args, "scheduler_type", "steplr")).lower()

    if scheduler_type == "steplr":
        return StepLR(
            optim,
            step_size=getattr(args, "steplr_step_size", 10),
            gamma=getattr(args, "steplr_gamma", 0.99),
        )

    if scheduler_type == "onecycle":
        return OneCycleLR(
            optim,
            max_lr=getattr(args, "learning_rate", 1e-4),
            total_steps=getattr(args, "tot_steps", 1000),
            div_factor=getattr(args, "ocyclr_div_factor", 25.0),
            pct_start=getattr(args, "ocyclr_pct_start", 0.3),
            anneal_strategy=getattr(args, "ocyclr_anneal_strategy", "cos"),
            final_div_factor=getattr(args, "ocyclr_final_div_factor", 1e4),
        )

    if scheduler_type == "reduceonplateau":
        return ReduceLROnPlateau(
            optim,
            mode="min",
            factor=getattr(args, "rpllr_factor", 0.95),
            patience=getattr(args, "rpllr_patience", 50),
            verbose=False,
            min_lr=getattr(args, "rpllr_min_lr", 0.0),
        )

    if scheduler_type == "noamlr":
        return NoamLR(
            optim,
            model_size=getattr(args, "hidden_nf", 64),
            warmup_steps=getattr(args, "warmup_step", 4000),
        )

    raise NotImplementedError(f"Unsupported scheduler_type: {scheduler_type}")


def wrap_model_for_training(model, args, distributed: bool, gpu: int):
    if distributed:
        return DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

    if bool(getattr(args, "dp", False)) and torch.cuda.device_count() > 1:
        model_dp = torch.nn.DataParallel(model.cpu())
        return model_dp.cuda()

    return model


def prepare_parallel_models(model, args, distributed: bool, gpu: int) -> ParallelModelBundle:
    model_dp = wrap_model_for_training(model, args, distributed=distributed, gpu=gpu)

    ema_decay = float(getattr(args, "ema_decay", 0.0))
    if ema_decay <= 0:
        return ParallelModelBundle(
            model_dp=model_dp,
            model_ema=model,
            model_ema_dp=model_dp,
            ema=None,
        )

    model_ema = deepcopy(model)
    ema = EMA(ema_decay)

    if distributed:
        model_ema_dp = DDP(model_ema, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    elif bool(getattr(args, "dp", False)) and torch.cuda.device_count() > 1:
        model_ema_dp = torch.nn.DataParallel(model_ema)
    else:
        model_ema_dp = model_ema

    return ParallelModelBundle(
        model_dp=model_dp,
        model_ema=model_ema,
        model_ema_dp=model_ema_dp,
        ema=ema,
    )
