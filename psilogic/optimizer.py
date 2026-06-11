"""
PsiLogic optimizer — Active Cancellation extension of Adam.

Extends Adam with a chaos-conditioned damping term that is strongest during
unstable early training and decays automatically at convergence.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

from ._chaos import (
    chaos_contribution,
    effective_gamma_and_qd,
    resolve_warmup,
    update_gradient_norm_ema,
)


def _validate_hyperparameters(
    lr: float,
    weight_decay: float,
    gamma: float,
    quantum_decay: float,
    betas: tuple[float, float],
    agc_clip: float,
    max_cancel: float,
) -> None:
    assert lr >= 0, f"Invalid lr: {lr}"
    assert weight_decay >= 0, f"Invalid weight_decay: {weight_decay}"
    assert gamma >= 0, f"Invalid gamma: {gamma}"
    assert quantum_decay >= 0, f"Invalid quantum_decay: {quantum_decay}"
    assert 0 <= betas[0] < 1, f"Invalid beta1: {betas[0]}"
    assert 0 <= betas[1] < 1, f"Invalid beta2: {betas[1]}"
    assert agc_clip >= 0, f"Invalid agc_clip: {agc_clip}"
    assert 0 < max_cancel <= 1, f"Invalid max_cancel: {max_cancel}"


def _apply_agc(grad: torch.Tensor, param: torch.Tensor, agc: float) -> torch.Tensor:
    if agc <= 0.0:
        return grad
    p_norm = param.norm()
    g_norm = grad.norm()
    max_norm = agc * p_norm.clamp(min=1e-3)
    clip_cf = (max_norm / g_norm.clamp(min=1e-6)).clamp(max=1.0)
    return grad * clip_cf


def _centralize_grad(grad: torch.Tensor) -> torch.Tensor:
    if grad.dim() <= 1:
        return grad
    return grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)


def _init_param_state(state: dict[str, Any], param: torch.Tensor) -> None:
    state["t"] = 0
    state["m"] = torch.zeros_like(param)
    state["v"] = torch.zeros_like(param)
    state["fast"] = torch.zeros(1, device=param.device, dtype=param.dtype)
    state["slow"] = torch.zeros(1, device=param.device, dtype=param.dtype)
    state["gn_avg"] = torch.zeros(1, device=param.device, dtype=param.dtype)


class PsiLogic(Optimizer):
    r"""
    Active Cancellation optimizer for deep neural networks.

    Combines Adam (or Lion sign-momentum) with a dual-EMA chaos detector that
    modulates per-step parameter shrinkage. See the project README for the
    full mathematical description.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate. Default: ``1e-3``.
        betas: Adam EMA coefficients ``(β₁, β₂)``. Default: ``(0.9, 0.999)``.
        weight_decay: Decoupled L₂ penalty (AdamW style). Default: ``1e-4``.
        gamma: Maximum active cancellation strength. Default: ``0.05``.
        p_ext: Chaos amplification factor. Default: ``1.0``.
        quantum_decay: Secondary decay coefficient; ``0.0`` disables. Default: ``0.0``.
        eps: Numerical stability epsilon. Default: ``1e-8``.
        grad_centralize: Subtract spatial mean from gradients. Default: ``True``.
        chaos_tau: Absolute slow-EMA threshold when ``adaptive_tau=False``.
        chaos_warmup: Warmup steps before chaos activates; ``-1`` auto-scales.
        adaptive_tau: Gate chaos on fast/slow ratio instead of absolute norm.
        tau_scale: Required fast/slow ratio in adaptive mode. Default: ``2.0``.
        max_cancel: Hard cap on per-step fractional shrinkage. Default: ``0.05``.
        agc_clip: Adaptive gradient clipping ratio; ``0.0`` disables.
        gamma_T_max: Total steps for cosine γ schedule; ``0`` disables.
        use_foreach: Batched CUDA ops via ``torch._foreach_*``. Default: ``True``.
        lion_mode: Sign-momentum (Lion) update instead of Adam. Default: ``False``.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-4,
        gamma: float = 0.05,
        p_ext: float = 1.0,
        quantum_decay: float = 0.0,
        eps: float = 1e-8,
        grad_centralize: bool = True,
        chaos_tau: float = 0.5,
        chaos_warmup: int = -1,
        adaptive_tau: bool = True,
        tau_scale: float = 2.0,
        max_cancel: float = 0.05,
        agc_clip: float = 0.02,
        gamma_T_max: int = 0,
        use_foreach: bool = True,
        lion_mode: bool = False,
    ) -> None:
        _validate_hyperparameters(
            lr, weight_decay, gamma, quantum_decay, betas, agc_clip, max_cancel
        )

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "p_ext": p_ext,
            "quantum_decay": quantum_decay,
            "eps": eps,
            "grad_centralize": grad_centralize,
            "chaos_tau": chaos_tau,
            "chaos_warmup": chaos_warmup,
            "adaptive_tau": adaptive_tau,
            "tau_scale": tau_scale,
            "max_cancel": max_cancel,
            "agc_clip": agc_clip,
            "gamma_T_max": gamma_T_max,
            "use_foreach": use_foreach,
            "lion_mode": lion_mode,
        }
        super().__init__(params, defaults)

    def _apply_unified_decay(
        self,
        param: torch.Tensor,
        raw_grad: torch.Tensor,
        *,
        lr: float,
        wd: float,
        gamma_eff: float,
        qd_eff: float,
        p_ext: float,
        max_cancel: float,
        slow_t: torch.Tensor,
        fast_t: torch.Tensor,
        adaptive_tau: bool,
        chaos_tau: float,
        tau_scale: float,
        eps: float,
        chaos_active: bool,
    ) -> None:
        if chaos_active and gamma_eff > 0:
            chaos_contrib, spike_mask = chaos_contribution(
                slow_t,
                fast_t,
                adaptive_tau=adaptive_tau,
                chaos_tau=chaos_tau,
                tau_scale=tau_scale,
                eps=eps,
                lr=lr,
                gamma_eff=gamma_eff,
                p_ext=p_ext,
                max_cancel=max_cancel,
                param_dtype=param.dtype,
            )
            total_scalar_decay = lr * wd + chaos_contrib
            param.mul_(1.0 - total_scalar_decay)

            if qd_eff > 0:
                qd_contrib = qd_eff * (1.0 - spike_mask)
                param.mul_(1.0 - lr * qd_contrib * torch.tanh(raw_grad.abs()))
        elif wd > 0:
            param.mul_(1.0 - lr * wd)

    def _adam_or_lion_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        step: int,
        lion: bool,
    ) -> None:
        if lion:
            update = (beta1 * state["m"] + (1.0 - beta1) * grad).sign()
            param.add_(update, alpha=-lr)
        else:
            bc1 = 1.0 - beta1**step
            bc2 = math.sqrt(1.0 - beta2**step)
            step_size = lr * bc2 / bc1
            denom = state["v"].sqrt().add_(eps)
            param.addcdiv_(state["m"], denom, value=-step_size)

    def _step_scalar(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        gamma = group["gamma"]
        p_ext = group["p_ext"]
        qd = group["quantum_decay"]
        eps = group["eps"]
        gc = group["grad_centralize"]
        chaos_tau = group["chaos_tau"]
        warmup = resolve_warmup(group["chaos_warmup"], group["gamma_T_max"])
        adapt_tau = group["adaptive_tau"]
        tau_scale = group["tau_scale"]
        max_cancel = group["max_cancel"]
        agc = group["agc_clip"]
        gamma_t_max = group["gamma_T_max"]
        lion = group["lion_mode"]

        for param in group["params"]:
            if param.grad is None:
                continue

            raw_grad = param.grad
            grad = _apply_agc(raw_grad, param, agc)
            if agc > 0.0:
                raw_grad = grad
            if gc:
                grad = _centralize_grad(grad)

            state = self.state[param]
            if not state:
                _init_param_state(state, param)

            state["t"] += 1
            step = state["t"]

            state["m"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
            if not lion:
                state["v"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            update_gradient_norm_ema(
                grad.norm(),
                grad.numel(),
                step,
                state["fast"],
                state["slow"],
                state["gn_avg"],
                eps,
            )

            gamma_eff, qd_eff = effective_gamma_and_qd(step, gamma_t_max, gamma, qd)
            chaos_active = step > warmup

            self._apply_unified_decay(
                param,
                raw_grad,
                lr=lr,
                wd=wd,
                gamma_eff=gamma_eff,
                qd_eff=qd_eff,
                p_ext=p_ext,
                max_cancel=max_cancel,
                slow_t=state["slow"],
                fast_t=state["fast"],
                adaptive_tau=adapt_tau,
                chaos_tau=chaos_tau,
                tau_scale=tau_scale,
                eps=eps,
                chaos_active=chaos_active,
            )

            self._adam_or_lion_update(
                param,
                grad,
                state,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                step=step,
                lion=lion,
            )

    def _step_foreach(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]
        gamma = group["gamma"]
        p_ext = group["p_ext"]
        qd = group["quantum_decay"]
        eps = group["eps"]
        gc = group["grad_centralize"]
        warmup = resolve_warmup(group["chaos_warmup"], group["gamma_T_max"])
        adapt_tau = group["adaptive_tau"]
        tau_scale = group["tau_scale"]
        max_cancel = group["max_cancel"]
        agc = group["agc_clip"]
        gamma_t_max = group["gamma_T_max"]
        lion = group["lion_mode"]

        params_with_grad = [p for p in group["params"] if p.grad is not None]
        if not params_with_grad:
            return

        grads = [p.grad for p in params_with_grad]

        if agc > 0.0:
            p_norms = torch._foreach_norm(params_with_grad)
            g_norms = torch._foreach_norm(grads)
            grads = [
                g * (agc * pn.clamp(min=1e-3) / gn.clamp(min=1e-6)).clamp(max=1.0)
                for g, pn, gn in zip(grads, p_norms, g_norms)
            ]

        raw_grads = [g.clone() for g in grads]

        if gc:
            for i, grad in enumerate(grads):
                grads[i] = _centralize_grad(grad)

        states = []
        for param in params_with_grad:
            state = self.state[param]
            if not state:
                _init_param_state(state, param)
            state["t"] += 1
            states.append(state)

        step = states[0]["t"]
        ms = [s["m"] for s in states]
        vs = [s["v"] for s in states]

        torch._foreach_mul_(ms, beta1)
        torch._foreach_add_(ms, grads, alpha=1.0 - beta1)
        if not lion:
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, grads, grads, value=1.0 - beta2)

        g_norms = torch._foreach_norm(grads)
        for gn, state, grad in zip(g_norms, states, grads):
            update_gradient_norm_ema(
                gn, grad.numel(), step, state["fast"], state["slow"], state["gn_avg"], eps
            )

        gamma_eff, qd_eff = effective_gamma_and_qd(step, gamma_t_max, gamma, qd)
        chaos_active = step > warmup

        if chaos_active and gamma_eff > 0:
            for param, raw_grad, state in zip(params_with_grad, raw_grads, states):
                self._apply_unified_decay(
                    param,
                    raw_grad,
                    lr=lr,
                    wd=wd,
                    gamma_eff=gamma_eff,
                    qd_eff=qd_eff,
                    p_ext=p_ext,
                    max_cancel=max_cancel,
                    slow_t=state["slow"],
                    fast_t=state["fast"],
                    adaptive_tau=adapt_tau,
                    chaos_tau=group["chaos_tau"],
                    tau_scale=tau_scale,
                    eps=eps,
                    chaos_active=True,
                )
        elif wd > 0:
            torch._foreach_mul_(params_with_grad, 1.0 - lr * wd)

        if lion:
            for param, momentum, grad in zip(params_with_grad, ms, grads):
                update = (beta1 * momentum + (1.0 - beta1) * grad).sign()
                param.add_(update, alpha=-lr)
        else:
            bc1 = 1.0 - beta1**step
            bc2 = math.sqrt(1.0 - beta2**step)
            step_size = lr * bc2 / bc1
            denoms = torch._foreach_sqrt(vs)
            torch._foreach_add_(denoms, eps)
            torch._foreach_addcdiv_(params_with_grad, ms, denoms, value=-step_size)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_foreach = group["use_foreach"] and any(
                p.is_cuda for p in group["params"] if p.grad is not None
            )
            if use_foreach:
                self._step_foreach(group)
            else:
                self._step_scalar(group)

        return loss
