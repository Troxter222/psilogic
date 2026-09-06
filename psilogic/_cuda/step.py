"""Orchestrate fused Triton steps for a PsiLogic param group."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from psilogic._chaos import (
    auto_gamma,
    auto_gamma_batched,
    effective_gamma_and_qd,
    effective_warmup,
    grad_momentum_disagreement,
    soft_chaos_signal,
    trust_from_soft_chaos,
    update_gradient_norm_ema,
    update_gradient_norm_ema_batched,
)
from psilogic.optimizer import (
    _apply_agc,
    _bind_packed_chaos_views,
    _centralize_grad,
    _chaos_views_match,
    _copy_chaos_into_packed,
    _init_param_state,
    _write_soft_chaos,
)

from . import kernels

_FUSED_BLOCK = 256

# Group metadata lives off the param-group dict so ``state_dict`` stays clean.
_GROUP_CUDA_CACHE: dict[int, dict[str, Any]] = {}

_ZERO_CACHE: dict[torch.device, torch.Tensor] = {}
_ONE_CACHE: dict[torch.device, torch.Tensor] = {}


def _group_cache(group: dict[str, Any]) -> dict[str, Any]:
    key = id(group)
    cache = _GROUP_CUDA_CACHE.get(key)
    if cache is None:
        cache = {}
        _GROUP_CUDA_CACHE[key] = cache
    return cache


def _maybe_contiguous(grad: torch.Tensor) -> torch.Tensor:
    return grad if grad.is_contiguous() else grad.contiguous()


def _zero_scalar(device: torch.device) -> torch.Tensor:
    z = _ZERO_CACHE.get(device)
    if z is None:
        z = torch.zeros(1, device=device, dtype=torch.float32)
        _ZERO_CACHE[device] = z
    return z


def _one_scalar(device: torch.device) -> torch.Tensor:
    """Neutral ``trust`` (no chaos damping) for this device."""
    o = _ONE_CACHE.get(device)
    if o is None:
        o = torch.ones(1, device=device, dtype=torch.float32)
        _ONE_CACHE[device] = o
    return o


def _ensure_packed_chaos(
    cache: dict[str, Any],
    states: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(states)
    packed = cache.get("packed")
    if packed is None or packed["n"] != n:
        packed_fast = torch.empty(n, device=device, dtype=torch.float32)
        packed_slow = torch.empty(n, device=device, dtype=torch.float32)
        packed_gn_avg = torch.empty(n, device=device, dtype=torch.float32)
        _copy_chaos_into_packed(states, packed_fast, packed_slow, packed_gn_avg)
        _bind_packed_chaos_views(states, packed_fast, packed_slow, packed_gn_avg)
        packed = {
            "n": n,
            "fast": packed_fast,
            "slow": packed_slow,
            "gn_avg": packed_gn_avg,
        }
        cache["packed"] = packed
    elif not _chaos_views_match(states, packed["fast"], packed["slow"], packed["gn_avg"]):
        _copy_chaos_into_packed(states, packed["fast"], packed["slow"], packed["gn_avg"])
        _bind_packed_chaos_views(states, packed["fast"], packed["slow"], packed["gn_avg"])
    return packed["fast"], packed["slow"], packed["gn_avg"]


def _cached_sqrt_numels(
    cache: dict[str, Any],
    grads: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    numels = [g.numel() for g in grads]
    if cache.get("grad_numels") != numels:
        cache["grad_numels"] = numels
        cache["sqrt_numels"] = torch.tensor(
            [math.sqrt(max(n, 1)) for n in numels],
            device=device,
            dtype=torch.float32,
        )
    return cache["sqrt_numels"]


def _ensure_multitensor_tables(
    cache: dict[str, Any],
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    raw_bufs: list[torch.Tensor],
    states: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return cached pointer/block tables, or None when the multi-tensor path cannot run."""
    n = len(params)
    if n == 0:
        return None
    param_dtype = params[0].dtype
    if any(g.dtype != param_dtype for g in grads) or any(
        rg.dtype != param_dtype for rg in raw_bufs
    ):
        return None
    if kernels._tl_param_dtype(param_dtype) is None:
        return None
    numels_list = [int(g.numel()) for g in grads]
    if any(nel > kernels._MAX_INT32 for nel in numels_list):
        return None

    device = params[0].device
    mt = cache.get("mt")
    p_sig = tuple(p.data_ptr() for p in params)
    m_sig = tuple(s["m"].data_ptr() for s in states)
    v_sig = tuple(s["v"].data_ptr() for s in states)

    need_blocks = mt is None or cache.get("mt_numels") != numels_list
    need_pmv = (
        need_blocks
        or cache.get("p_sig") != p_sig
        or cache.get("m_sig") != m_sig
        or cache.get("v_sig") != v_sig
    )

    if need_blocks:
        tensor_ids: list[int] = []
        block_offs: list[int] = []
        for i, nel in enumerate(numels_list):
            nblocks = (nel + _FUSED_BLOCK - 1) // _FUSED_BLOCK
            for b in range(nblocks):
                off = b * _FUSED_BLOCK
                if off > kernels._MAX_INT32:
                    return None
                tensor_ids.append(i)
                block_offs.append(b)
        total_blocks = len(tensor_ids)
        if total_blocks == 0:
            tensor_ids = [0]
            block_offs = [0]
        mt = {
            "numels": torch.tensor(numels_list, device=device, dtype=torch.int32),
            "tensor_id": torch.tensor(tensor_ids, device=device, dtype=torch.int32),
            "block_off": torch.tensor(block_offs, device=device, dtype=torch.int32),
            "total_blocks": total_blocks,
            "param_dtype": param_dtype,
            "g_host": torch.empty(n, dtype=torch.int64, device="cpu"),
            "raw_host": torch.empty(n, dtype=torch.int64, device="cpu"),
        }
        cache["mt"] = mt
        cache["mt_numels"] = numels_list

    if need_pmv:
        mt["p_ptrs"] = torch.tensor(list(p_sig), device=device, dtype=torch.int64)
        mt["m_ptrs"] = torch.tensor(list(m_sig), device=device, dtype=torch.int64)
        mt["v_ptrs"] = torch.tensor(list(v_sig), device=device, dtype=torch.int64)
        cache["p_sig"] = p_sig
        cache["m_sig"] = m_sig
        cache["v_sig"] = v_sig

    g_host = mt["g_host"]
    raw_host = mt["raw_host"]
    for i, g in enumerate(grads):
        g_host[i] = g.data_ptr()
    for i, rg in enumerate(raw_bufs):
        raw_host[i] = rg.data_ptr()
    if "grad_ptrs" not in mt:
        mt["grad_ptrs"] = g_host.to(device=device)
        mt["raw_ptrs"] = raw_host.to(device=device)
    else:
        mt["grad_ptrs"].copy_(g_host, non_blocking=True)
        mt["raw_ptrs"].copy_(raw_host, non_blocking=True)
    return mt


def _launch_per_tensor_fused(
    params_with_grad: list[torch.Tensor],
    states: list[dict[str, Any]],
    grads: list[torch.Tensor],
    raw_bufs: list[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
    step_size: float,
    wd_decay: float,
    trust_vec: torch.Tensor,
    qd_contrib_vec: torch.Tensor,
    apply_quantum: bool,
    lion: bool,
    device: torch.device,
) -> None:
    dummy_leader_sum = _zero_scalar(device)
    for i, param in enumerate(params_with_grad):
        kernels.launch_fused_step(
            grads[i],
            raw_bufs[i],
            param,
            states[i]["m"],
            states[i]["v"],
            grad_centralize=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            eps=eps,
            step_size=step_size,
            wd_decay=wd_decay,
            trust=trust_vec[i : i + 1],
            qd_contrib=qd_contrib_vec[i : i + 1],
            apply_quantum=apply_quantum,
            lion=lion,
            n_leaders=1,
            elems_per_leader=1,
            leader_sum=dummy_leader_sum,
        )


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
    prepared: bool = False,
) -> None:
    """Fused CUDA step for one parameter tensor (matches ``_step_scalar`` order)."""
    if param.grad is None:
        return

    raw_grad = param.grad
    grad = _apply_agc(raw_grad, param, agc)
    if agc > 0.0:
        raw_grad = grad

    if not prepared:
        if not state:
            _init_param_state(state, param)
        state["t"] += 1
    step = state["t"]

    raw_grad = _maybe_contiguous(raw_grad)
    grad = _maybe_contiguous(grad)
    if gc:
        grad = _maybe_contiguous(_centralize_grad(grad))

    g_norm = grad.norm()

    # Disagreement must be read while ``m`` still holds the previous step —
    # the fused kernel updates momentum in-place.
    gamma_sched, _ = effective_gamma_and_qd(step, gamma_t_max, gamma, qd)
    chaos_gain = effective_warmup(step, gamma_t_max, warmup_cfg)
    disagree = None
    if chaos_gain > 0.0 and gamma_sched > 0:
        disagree = grad_momentum_disagreement(
            grad, state["m"], g_norm, step=step, eps=eps
        )

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

    trust = _one_scalar(param.device)
    qd_contrib = _zero_scalar(param.device)
    apply_quantum = False
    wd_decay = lr * wd if wd > 0 else 0.0

    if disagree is not None and chaos_gain > 0.0 and gamma_eff > 0:
        soft_chaos = soft_chaos_signal(
            state["slow"],
            state["fast"],
            disagree,
            adaptive_tau=adapt_tau,
            chaos_tau=chaos_tau,
            tau_scale=tau_scale,
            eps=eps,
        )
        state["soft_chaos"].copy_(soft_chaos)
        trust = trust_from_soft_chaos(
            soft_chaos,
            gamma_eff=gamma_eff,
            p_ext=p_ext,
            chaos_gain=chaos_gain,
            max_cancel=max_cancel,
        )
        if qd_eff > 0:
            qd_contrib = (qd_eff * chaos_gain * (1.0 - soft_chaos)).reshape(1).to(torch.float32)
            apply_quantum = True
    else:
        state["soft_chaos"].zero_()

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
        wd_decay=wd_decay,
        trust=trust,
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

    step = states[0]["t"]
    uniform_step = all(s["t"] == step for s in states)
    homogeneous = len({(p.device, p.dtype) for p in params_with_grad}) == 1

    if not (uniform_step and homogeneous):
        for param, state in zip(params_with_grad, states):
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
                prepared=True,
            )
        return

    grads = [p.grad for p in params_with_grad]
    if agc > 0.0:
        p_norms = torch.stack(torch._foreach_norm(params_with_grad))
        g_norms = torch.stack(torch._foreach_norm(grads))
        max_norms = agc * p_norms.clamp(min=1e-3)
        clip_factors = (max_norms / g_norms.clamp(min=1e-6)).clamp(max=1.0)
        torch._foreach_mul_(grads, clip_factors.unbind())

    raw_grads_buf = [_maybe_contiguous(g) for g in grads]
    if gc:
        grads = [
            _maybe_contiguous(_centralize_grad(g)) if g.dim() > 1 else _maybe_contiguous(g)
            for g in grads
        ]
    else:
        grads = [_maybe_contiguous(g) for g in grads]

    g_norms = torch.stack(torch._foreach_norm(grads))

    gamma_eff, qd_eff = effective_gamma_and_qd(step, gamma_t_max, gamma, qd)
    chaos_gain = effective_warmup(step, gamma_t_max, warmup_cfg)

    # Grad-vs-momentum disagreement before the fused kernel mutates ``m``.
    disagrees: list[torch.Tensor] = []
    if chaos_gain > 0.0 and gamma_eff > 0:
        disagrees = [
            grad_momentum_disagreement(
                grad, state["m"], g_norms[i], step=state["t"], eps=eps
            )
            for i, (state, grad) in enumerate(zip(states, grads))
        ]

    cache = _group_cache(group)
    param_ids = tuple(id(p) for p in params_with_grad)
    if cache.get("param_ids") != param_ids:
        cache.clear()
        cache["param_ids"] = param_ids

    fast_vec, slow_vec, gn_avg_vec = _ensure_packed_chaos(cache, states, params_with_grad[0].device)
    sqrt_numels = _cached_sqrt_numels(cache, grads, fast_vec.device)
    gn_scaled_vec = g_norms / sqrt_numels

    update_gradient_norm_ema_batched(gn_scaled_vec, step, fast_vec, slow_vec, gn_avg_vec, eps)

    # Packed views write through: maybe_sync updates fast_vec/slow_vec in place.
    maybe_sync(states)

    gamma_eff_vec: torch.Tensor | float
    if gamma_auto_on:
        gamma_eff_vec = auto_gamma_batched(slow_vec, step, gamma_eff)
    else:
        gamma_eff_vec = float(gamma_eff)

    chaos_gain = effective_warmup(step, gamma_t_max, warmup_cfg)

    dev = params_with_grad[0].device
    n_states = len(states)
    wd_decay = lr * wd if wd > 0 else 0.0

    if chaos_gain > 0.0 and (isinstance(gamma_eff_vec, torch.Tensor) or gamma_eff_vec > 0) and disagrees:
        soft_vec = soft_chaos_signal(
            slow_vec,
            fast_vec,
            torch.cat(disagrees),
            adaptive_tau=adapt_tau,
            chaos_tau=chaos_tau,
            tau_scale=tau_scale,
            eps=eps,
        )
        _write_soft_chaos(states, soft_vec)
        trust_vec = trust_from_soft_chaos(
            soft_vec,
            gamma_eff=gamma_eff_vec,
            p_ext=p_ext,
            chaos_gain=chaos_gain,
            max_cancel=max_cancel,
        ).to(torch.float32).contiguous()

        if qd_eff > 0:
            qd_contrib_vec = (
                (qd_eff * chaos_gain * (1.0 - soft_vec)).to(torch.float32).contiguous()
            )
            apply_quantum = True
        else:
            qd_contrib_vec = torch.zeros(n_states, device=dev, dtype=torch.float32)
            apply_quantum = False
    else:
        _write_soft_chaos(states, None)
        trust_vec = torch.ones(n_states, device=dev, dtype=torch.float32)
        qd_contrib_vec = torch.zeros(n_states, device=dev, dtype=torch.float32)
        apply_quantum = False

    if lion:
        step_size = lr
    else:
        bc1 = 1.0 - beta1**step
        bc2 = math.sqrt(1.0 - beta2**step)
        step_size = lr * bc2 / bc1

    mt = _ensure_multitensor_tables(cache, params_with_grad, grads, raw_grads_buf, states)
    launched = False
    if mt is not None:
        launched = kernels.launch_multi_fused_step(
            mt["grad_ptrs"],
            mt["raw_ptrs"],
            mt["p_ptrs"],
            mt["m_ptrs"],
            mt["v_ptrs"],
            mt["numels"],
            mt["tensor_id"],
            mt["block_off"],
            mt["total_blocks"],
            wd_decay,
            trust_vec,
            qd_contrib_vec,
            apply_quantum=apply_quantum,
            lion=lion,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            eps=eps,
            step_size=step_size,
            param_dtype=mt["param_dtype"],
        )
    if not launched:
        _launch_per_tensor_fused(
            params_with_grad,
            states,
            grads,
            raw_grads_buf,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            eps=eps,
            step_size=step_size,
            wd_decay=wd_decay,
            trust_vec=trust_vec,
            qd_contrib_vec=qd_contrib_vec,
            apply_quantum=apply_quantum,
            lion=lion,
            device=dev,
        )
