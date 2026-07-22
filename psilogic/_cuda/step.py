"""Orchestrate fused Triton steps for a PsiLogic param group."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from psilogic._chaos import (
    auto_gamma,
    chaos_contribution,
    effective_gamma_and_qd,
    effective_warmup,
    update_gradient_norm_ema,
)
from psilogic.optimizer import _apply_agc, _centralize_grad, _init_param_state

from . import kernels


def _leader_layout(param: torch.Tensor) -> tuple[int, int]:
    if param.dim() <= 1:
        return 1, param.numel()
    n_leaders = param.shape[0]
    return n_leaders, param.numel() // n_leaders


def fused_param_step(
    param: torch.Tensor,
    state: dict[str, Any],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    wd: float,
    gamma: float,
    p_ext: float,
    qd: float,
    eps: float,
    gc: bool,
    chaos_tau: float,
    warmup_cfg: int,
    adapt_tau: bool,
    tau_scale: float,
    max_cancel: float,
    agc: float,
    gamma_t_max: int,
    lion: bool,
    gamma_auto_on: bool,
    maybe_sync: Callable[[list[dict[str, Any]]], None],
) -> None:
    """Fused CUDA step for one parameter tensor (matches ``_step_scalar`` order)."""
    if param.grad is None:
        return

    raw_grad = param.grad
    grad = _apply_agc(raw_grad, param, agc)
    if agc > 0.0:
        raw_grad = grad

    if not state:
        _init_param_state(state, param)

    state["t"] += 1
    step = state["t"]

    raw_grad = raw_grad.contiguous()
    grad = grad.contiguous()
    if gc:
        # Use the scalar reference reduction here.  Triton's atomic row sums
        # are order-dependent and can move the chaos EMA enough to alter later
        # cancellation decisions, particularly for long FP32 runs.
        grad = _centralize_grad(grad).contiguous()
    raw_grad_buf = torch.empty_like(grad)
    n_leaders, elems_per_leader = _leader_layout(param)
    leader_sum = torch.zeros(n_leaders, device=param.device, dtype=torch.float32)

    kernels.launch_centralize_moment(
        grad,
        raw_grad,
        raw_grad_buf,
        state["m"],
        state["v"],
        grad_centralize=False,
        beta1=beta1,
        beta2=beta2,
        update_variance=not lion,
        update_momentum=not lion,
        n_leaders=n_leaders,
        elems_per_leader=elems_per_leader,
        leader_sum=leader_sum,
    )

    g_norm = grad.norm()
    update_gradient_norm_ema(
        g_norm,
        grad.numel(),
        step,
        state["fast"],
        state["slow"],
        state["gn_avg"],
        eps,
    )
    maybe_sync([state])

    gamma_eff, qd_eff = effective_gamma_and_qd(step, gamma_t_max, gamma, qd)
    if gamma_auto_on:
        gamma_eff = auto_gamma(state["slow"], step, gamma_eff)
    chaos_gain = effective_warmup(step, gamma_t_max, warmup_cfg)

    total_scalar_decay = 0.0
    wd_only_decay = 0.0
    qd_contrib = 0.0
    apply_quantum = False

    if chaos_gain > 0.0 and gamma_eff > 0:
        chaos_contrib, spike_mask = chaos_contribution(
            state["slow"],
            state["fast"],
            adaptive_tau=adapt_tau,
            chaos_tau=chaos_tau,
            tau_scale=tau_scale,
            eps=eps,
            lr=lr,
            gamma_eff=gamma_eff,
            p_ext=p_ext,
            max_cancel=max_cancel,
            param_dtype=param.dtype,
        )
        total_scalar_decay = float((lr * wd + chaos_contrib * chaos_gain).item())
        if qd_eff > 0:
            qd_contrib = float((qd_eff * chaos_gain * (1.0 - spike_mask)).item())
            apply_quantum = True
    elif wd > 0:
        wd_only_decay = lr * wd

    if lion:
        step_size = lr
    else:
        bc1 = 1.0 - beta1**step
        bc2 = math.sqrt(1.0 - beta2**step)
        step_size = lr * bc2 / bc1

    kernels.launch_decay_adam(
        grad,
        raw_grad_buf,
        param,
        state["m"],
        state["v"],
        lr=lr,
        eps=eps,
        step_size=step_size,
        total_scalar_decay=total_scalar_decay,
        wd_only_decay=wd_only_decay,
        qd_contrib=qd_contrib,
        apply_quantum=apply_quantum,
        lion=lion,
        beta1=beta1,
        beta2=beta2,
    )


def fused_group_step(
    group: dict[str, Any],
    state_dict: dict[Any, dict[str, Any]],
    *,
    sync_chaos_ddp: bool,
    maybe_sync: Callable[[list[dict[str, Any]]], None],
) -> None:
    """Run fused steps for all parameters in a group."""
    lr = group["lr"]
    beta1, beta2 = group["betas"]
    wd = group["weight_decay"]
    gamma = group["gamma"]
    p_ext = group["p_ext"]
    qd = group["quantum_decay"]
    eps = group["eps"]
    gc = group["grad_centralize"]
    chaos_tau = group["chaos_tau"]
    warmup_cfg = group["chaos_warmup"]
    adapt_tau = group["adaptive_tau"]
    tau_scale = group["tau_scale"]
    max_cancel = group["max_cancel"]
    agc = group["agc_clip"]
    gamma_t_max = group["gamma_T_max"]
    lion = group["lion_mode"]
    gamma_auto_on = group["gamma_auto"]

    for param in group["params"]:
        state = state_dict[param]
        fused_param_step(
            param,
            state,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            wd=wd,
            gamma=gamma,
            p_ext=p_ext,
            qd=qd,
            eps=eps,
            gc=gc,
            chaos_tau=chaos_tau,
            warmup_cfg=warmup_cfg,
            adapt_tau=adapt_tau,
            tau_scale=tau_scale,
            max_cancel=max_cancel,
            agc=agc,
            gamma_t_max=gamma_t_max,
            lion=lion,
            gamma_auto_on=gamma_auto_on,
            maybe_sync=maybe_sync,
        )
