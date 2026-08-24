"""Orchestrate fused Triton steps for a PsiLogic param group."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from psilogic._chaos import (
    auto_gamma,
    auto_gamma_batched,
    chaos_contribution,
    chaos_contribution_batched,
    effective_gamma_and_qd,
    effective_warmup,
    update_gradient_norm_ema,
    update_gradient_norm_ema_batched,
)
from psilogic.optimizer import _apply_agc, _centralize_grad, _init_param_state

from . import kernels

_FOREACH_COPY_AVAILABLE = hasattr(torch, "_foreach_copy_")

_ZERO_CACHE: dict[torch.device, torch.Tensor] = {}


def _zero_scalar(device: torch.device) -> torch.Tensor:
    z = _ZERO_CACHE.get(device)
    if z is None:
        z = torch.zeros(1, device=device, dtype=torch.float32)
        _ZERO_CACHE[device] = z
    return z


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
        grad = _centralize_grad(grad).contiguous()

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

    zero = _zero_scalar(param.device)
    total_scalar_decay = zero
    wd_only_decay = zero
    qd_contrib = zero
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
        total_scalar_decay = (
            (lr * wd + chaos_contrib * chaos_gain).reshape(1).to(torch.float32).contiguous()
        )
        if qd_eff > 0:
            qd_contrib = (
                (qd_eff * chaos_gain * (1.0 - spike_mask)).reshape(1).to(torch.float32).contiguous()
            )
            apply_quantum = True
    elif wd > 0:
        wd_only_decay = torch.full((1,), lr * wd, device=param.device, dtype=torch.float32)

    if lion:
        step_size = lr
    else:
        bc1 = 1.0 - beta1**step
        bc2 = math.sqrt(1.0 - beta2**step)
        step_size = lr * bc2 / bc1

    kernels.launch_fused_step(
        grad,
        raw_grad,
        param,
        state["m"],
        state["v"],
        grad_centralize=False,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        eps=eps,
        step_size=step_size,
        total_scalar_decay=total_scalar_decay,
        wd_only_decay=wd_only_decay,
        qd_contrib=qd_contrib,
        apply_quantum=apply_quantum,
        lion=lion,
        n_leaders=1,
        elems_per_leader=1,
        leader_sum=_zero_scalar(param.device),
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

    params_with_grad = [p for p in group["params"] if p.grad is not None]
    if not params_with_grad:
        return

    states = []
    for param in params_with_grad:
        state = state_dict[param]
        if not state:
            _init_param_state(state, param)
        state["t"] += 1
        states.append(state)

    raw_grads = [p.grad for p in params_with_grad]

    if agc > 0.0:
        p_norms = torch.stack(torch._foreach_norm(params_with_grad))
        g_norms = torch.stack(torch._foreach_norm(raw_grads))
        max_norms = agc * p_norms.clamp(min=1e-3)
        clip_factors = (max_norms / g_norms.clamp(min=1e-6)).clamp(max=1.0)
        grads = [g * cf for g, cf in zip(raw_grads, clip_factors.unbind())]
    else:
        grads = list(raw_grads)

    raw_grads_buf = [g.contiguous() for g in grads]

    if gc:
        grads = [_centralize_grad(g).contiguous() if g.dim() > 1 else g.contiguous() for g in grads]
    else:
        grads = [g.contiguous() for g in grads]

    g_norms = torch.stack(torch._foreach_norm(grads))

    step = states[0]["t"]
    uniform_step = all(s["t"] == step for s in states)
    homogeneous = len({(p.device, p.dtype) for p in params_with_grad}) == 1

    if uniform_step and homogeneous:
        fast_vec = torch.cat([s["fast"] for s in states])
        slow_vec = torch.cat([s["slow"] for s in states])
        gn_avg_vec = torch.cat([s["gn_avg"] for s in states])

        sqrt_numels = torch.tensor(
            [math.sqrt(max(g.numel(), 1)) for g in grads],
            device=fast_vec.device,
            dtype=fast_vec.dtype,
        )
        gn_scaled_vec = g_norms / sqrt_numels

        update_gradient_norm_ema_batched(gn_scaled_vec, step, fast_vec, slow_vec, gn_avg_vec, eps)

        if _FOREACH_COPY_AVAILABLE:
            fast_slices = [fast_vec[i : i + 1] for i in range(len(states))]
            slow_slices = [slow_vec[i : i + 1] for i in range(len(states))]
            gn_avg_slices = [gn_avg_vec[i : i + 1] for i in range(len(states))]
            torch._foreach_copy_([s["fast"] for s in states], fast_slices)
            torch._foreach_copy_([s["slow"] for s in states], slow_slices)
            torch._foreach_copy_([s["gn_avg"] for s in states], gn_avg_slices)
        else:
            for i, s in enumerate(states):
                s["fast"].copy_(fast_vec[i : i + 1])
                s["slow"].copy_(slow_vec[i : i + 1])
                s["gn_avg"].copy_(gn_avg_vec[i : i + 1])

        # One batched sync call for the whole group (matches the scalar
        # reference path's "N single-state calls" coverage, just issued as
        # a single call over all N states here) instead of the old inline
        # dist.all_reduce, which operated on `fast_vec`/`slow_vec` copies
        # and therefore never actually reached `maybe_sync` — DDP callers
        # relying on that callback got silently skipped in this path.
        # `maybe_sync` is a no-op when sync_chaos_ddp is False or no process
        # group is initialized, so it's safe to call unconditionally here,
        # matching `fused_param_step`'s call convention below.
        maybe_sync(states)
        if sync_chaos_ddp:
            # states' fast/slow tensors may have just been averaged across
            # ranks in place; refresh the local vectors so the chaos gate
            # computed below uses the synced values rather than this rank's
            # pre-sync copy.
            fast_vec = torch.cat([s["fast"] for s in states])
            slow_vec = torch.cat([s["slow"] for s in states])

        gamma_eff, qd_eff = effective_gamma_and_qd(step, gamma_t_max, gamma, qd)
        gamma_eff_vec: torch.Tensor | float
        if gamma_auto_on:
            gamma_eff_vec = auto_gamma_batched(slow_vec, step, gamma_eff)
        else:
            gamma_eff_vec = float(gamma_eff)

        chaos_gain = effective_warmup(step, gamma_t_max, warmup_cfg)

        dev = params_with_grad[0].device
        if chaos_gain > 0.0 and (isinstance(gamma_eff_vec, torch.Tensor) or gamma_eff_vec > 0):
            chaos_contrib_vec, spike_mask_vec = chaos_contribution_batched(
                slow_vec,
                fast_vec,
                adaptive_tau=adapt_tau,
                chaos_tau=chaos_tau,
                tau_scale=tau_scale,
                eps=eps,
                lr=lr,
                gamma_eff=gamma_eff_vec,
                p_ext=p_ext,
                max_cancel=max_cancel,
                param_dtype=fast_vec.dtype,
            )
            total_scalar_decay_vec = (
                (lr * wd + chaos_contrib_vec * chaos_gain).to(torch.float32).contiguous()
            )
            wd_only_decay_vec = torch.zeros(len(states), device=dev, dtype=torch.float32)

            if qd_eff > 0:
                qd_contrib_vec = (
                    (qd_eff * chaos_gain * (1.0 - spike_mask_vec)).to(torch.float32).contiguous()
                )
                apply_quantum = True
            else:
                qd_contrib_vec = torch.zeros(len(states), device=dev, dtype=torch.float32)
                apply_quantum = False
        elif wd > 0:
            total_scalar_decay_vec = torch.zeros(len(states), device=dev, dtype=torch.float32)
            wd_only_decay_vec = torch.full((len(states),), lr * wd, device=dev, dtype=torch.float32)
            qd_contrib_vec = torch.zeros(len(states), device=dev, dtype=torch.float32)
            apply_quantum = False
        else:
            total_scalar_decay_vec = torch.zeros(len(states), device=dev, dtype=torch.float32)
            wd_only_decay_vec = torch.zeros(len(states), device=dev, dtype=torch.float32)
            qd_contrib_vec = torch.zeros(len(states), device=dev, dtype=torch.float32)
            apply_quantum = False

        if lion:
            step_size = lr
        else:
            bc1 = 1.0 - beta1**step
            bc2 = math.sqrt(1.0 - beta2**step)
            step_size = lr * bc2 / bc1

        # grad_centralize is always False below (centralization was already
        # applied per-tensor via _centralize_grad above, exactly like the
        # foreach path) so the kernel's leader_sum branch is never taken -
        # a real per-tensor n_leaders-sized buffer here would just be a
        # wasted allocation + zero-fill kernel every step. The shared cached
        # dummy tensor is never read in this call, only its (unused) pointer
        # needs to be valid.
        dummy_leader_sum = _zero_scalar(params_with_grad[0].device)
        for i, param in enumerate(params_with_grad):
            state = states[i]
            grad = grads[i]
            raw_gbuf = raw_grads_buf[i]

            kernels.launch_centralize_moment(
                grad,
                raw_gbuf,
                raw_gbuf,
                state["m"],
                state["v"],
                grad_centralize=False,
                beta1=beta1,
                beta2=beta2,
                update_variance=not lion,
                update_momentum=not lion,
                n_leaders=1,
                elems_per_leader=1,
                leader_sum=dummy_leader_sum,
            )

            kernels.launch_decay_adam(
                grad,
                raw_gbuf,
                param,
                state["m"],
                state["v"],
                lr=lr,
                eps=eps,
                step_size=step_size,
                total_scalar_decay=total_scalar_decay_vec[i : i + 1],
                wd_only_decay=wd_only_decay_vec[i : i + 1],
                qd_contrib=qd_contrib_vec[i : i + 1],
                apply_quantum=apply_quantum,
                lion=lion,
                beta1=beta1,
                beta2=beta2,
            )
    else:
        for i, param in enumerate(params_with_grad):
            fused_param_step(
                param,
                states[i],
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
