"""
PsiLogic optimizer — Active Cancellation extension of Adam.

Extends Adam with a chaos-conditioned damping term that is strongest during
unstable early training and decays automatically at convergence.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ._chaos import (
    auto_gamma,
    auto_gamma_batched,
    chaos_contribution,
    chaos_contribution_batched,
    effective_gamma_and_qd,
    effective_warmup,
    update_gradient_norm_ema,
    update_gradient_norm_ema_batched,
)

_STATE_DICT_SCHEMA_KEY = "psilogic_schema"
_STATE_DICT_SCHEMA_VERSION = 2

_FOREACH_OPS = ("mul_", "add_", "addcmul_", "sqrt", "addcdiv_", "norm")
_FOREACH_AVAILABLE = all(hasattr(torch, f"_foreach_{op}") for op in _FOREACH_OPS)
_FOREACH_COPY_AVAILABLE = hasattr(torch, "_foreach_copy_")

# Cached per-(device, dtype) constant-1.0 scalar tensors. These stand in for
# a per-parameter "no-op" multiplier (chaos/quantum-decay disabled for this
# param this step) inside torch._foreach_mul_ calls, which only ever *read*
# the scale tensors. Reusing one shared tensor per (device, dtype) instead of
# allocating a fresh `torch.tensor(1.0, ...)` for every parameter, every
# step, removes a large number of small CUDA allocations/kernel launches
# without changing any computed value.
_ONE_SCALAR_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _one_scalar(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    t = _ONE_SCALAR_CACHE.get(key)
    if t is None:
        t = torch.ones(1, device=device, dtype=dtype)
        _ONE_SCALAR_CACHE[key] = t
    return t


def _validate_hyperparameters(
    lr: float,
    weight_decay: float,
    gamma: float,
    quantum_decay: float,
    betas: tuple[float, float],
    agc_clip: float,
    max_cancel: float,
) -> None:
    if lr < 0:
        raise ValueError(f"Invalid lr: {lr} (must be >= 0)")
    if weight_decay < 0:
        raise ValueError(f"Invalid weight_decay: {weight_decay} (must be >= 0)")
    if gamma < 0:
        raise ValueError(f"Invalid gamma: {gamma} (must be >= 0)")
    if quantum_decay < 0:
        raise ValueError(f"Invalid quantum_decay: {quantum_decay} (must be >= 0)")
    if not 0 <= betas[0] < 1:
        raise ValueError(f"Invalid beta1: {betas[0]} (must be in [0, 1))")
    if not 0 <= betas[1] < 1:
        raise ValueError(f"Invalid beta2: {betas[1]} (must be in [0, 1))")
    if agc_clip < 0:
        raise ValueError(f"Invalid agc_clip: {agc_clip} (must be >= 0)")
    if not 0 < max_cancel <= 1:
        raise ValueError(f"Invalid max_cancel: {max_cancel} (must be in (0, 1])")


def _apply_agc(grad: torch.Tensor, param: torch.Tensor, agc: float) -> torch.Tensor:
    if agc <= 0.0:
        return grad
    p_norm = param.norm()
    g_norm = grad.norm()
    max_norm = agc * p_norm.clamp(min=1e-3)
    clip_cf = (max_norm / g_norm.clamp(min=1e-6)).clamp(max=1.0)
    out: torch.Tensor = grad * clip_cf
    return out


def _centralize_grad(grad: torch.Tensor) -> torch.Tensor:
    if grad.dim() <= 1:
        return grad
    return grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)


def _init_param_state(state: dict[str, Any], param: torch.Tensor) -> None:
    # Optimizer state is always kept in fp32, even when the parameter itself
    # is fp16/bf16. Accumulating momentum/variance and the chaos EMAs at low
    # precision underflows and drifts to NaN over many steps; fp32 state is
    # the standard mixed-precision-optimizer pattern (as in fused Adam).
    state["t"] = 0
    state["m"] = torch.zeros_like(param, dtype=torch.float32)
    state["v"] = torch.zeros_like(param, dtype=torch.float32)
    state["fast"] = torch.zeros(1, device=param.device, dtype=torch.float32)
    state["slow"] = torch.zeros(1, device=param.device, dtype=torch.float32)
    state["gn_avg"] = torch.zeros(1, device=param.device, dtype=torch.float32)


class PsiLogic(Optimizer):
    r"""
    Active Cancellation optimizer for deep neural networks.

    Combines Adam (or Lion sign-momentum) with a dual-EMA chaos detector that
    modulates per-step parameter shrinkage. See the project README for the
    full mathematical description.

    All chaos hyperparameters are valid per param-group overrides, including
    ``lion_mode`` — e.g. ViT transformer blocks can run Lion updates while
    patch embeddings stay on Adam (see ``vit_param_groups(lion_blocks=True)``).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate. Default: ``1e-3``.
        betas: Adam EMA coefficients ``(β₁, β₂)``. Default: ``(0.9, 0.999)``.
        weight_decay: Decoupled L₂ penalty (AdamW style). Default: ``1e-4``.
        gamma: Maximum active cancellation strength. Default: ``0.05``.
        p_ext: Chaos amplification factor. Default: ``1.0``.
        quantum_decay: Secondary decay coefficient; ``0.0`` disables. Default: ``0.0``.
        eps: Numerical stability epsilon. Default: ``1e-8``.
        grad_centralize: Subtract spatial mean from gradients. Default: ``False``
            (plain drop-in). Opt in with ``True``, or use a task preset such as
            ``PsiLogicViT`` / ``vision_defaults``.
        chaos_tau: Absolute slow-EMA threshold when ``adaptive_tau=False``.
        chaos_warmup: Warmup steps before chaos activates; ``-1`` auto-scales
            to ``max(500, gamma_T_max // 20)``. Chaos then ramps in linearly
            over a quarter of the warmup window instead of switching on hard.
        adaptive_tau: Gate chaos on fast/slow ratio instead of absolute norm.
        tau_scale: Required fast/slow ratio in adaptive mode. Default: ``2.0``.
        max_cancel: Hard cap on per-step fractional shrinkage. Default: ``0.05``.
        agc_clip: Adaptive gradient clipping ratio; ``0.0`` disables (default).
            Task presets may enable a small clip (e.g. ViT).
        gamma_T_max: Total steps for cosine γ schedule; ``0`` disables.
        use_foreach: Batched CUDA ops via ``torch._foreach_*``. Falls back to
            the scalar path automatically when foreach ops are unavailable.
        use_fused_cuda: Triton fused CUDA step when CUDA and Triton are available.
            Falls back to ``use_foreach`` then scalar automatically.
        lion_mode: Sign-momentum (Lion) update instead of Adam. Default: ``False``.
        gamma_auto: Auto-reduce γ when the slow EMA signals convergence
            (``slow < 0.1``). Default: ``False``.
        sync_chaos_ddp: All-reduce (mean) the fast/slow chaos signals across
            DDP ranks before applying cancellation, so every rank damps
            identically. No-op outside an initialized process group.
        profile_step_time: Record wall-clock ``step()`` duration in
            ``self.last_step_time_ms`` and an EMA in ``self.step_time_ms_ema``.
            Timing is host-side and approximate under CUDA async execution.
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
        grad_centralize: bool = False,
        chaos_tau: float = 0.5,
        chaos_warmup: int = -1,
        adaptive_tau: bool = True,
        tau_scale: float = 2.0,
        max_cancel: float = 0.05,
        agc_clip: float = 0.0,
        gamma_T_max: int = 0,
        use_foreach: bool = True,
        use_fused_cuda: bool = True,
        lion_mode: bool = False,
        gamma_auto: bool = False,
        sync_chaos_ddp: bool = False,
        profile_step_time: bool = False,
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
            "use_fused_cuda": use_fused_cuda,
            "lion_mode": lion_mode,
            "gamma_auto": gamma_auto,
        }
        super().__init__(params, defaults)

        self._use_fused_cuda = bool(use_fused_cuda)
        self._sync_chaos_ddp = bool(sync_chaos_ddp)
        self._profile_step_time = bool(profile_step_time)
        self.last_step_time_ms: float = 0.0
        self.step_time_ms_ema: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Zero-config construction
    # ------------------------------------------------------------------ #

    @classmethod
    def auto(
        cls,
        model: nn.Module,
        lr: Optional[float] = None,
        total_steps: int = 0,
        **overrides: Any,
    ) -> PsiLogic:
        """Zero-config constructor: infer preset and param groups from ``model``.

        Architecture is detected from module types and parameter names
        (ViT / GPT / NLP encoder / CNN / generic) and the matching preset and
        parameter-group builder are applied automatically::

            optimizer = PsiLogic.auto(model, total_steps=len(loader) * epochs)
        """
        from .convenience import build_auto_optimizer

        return build_auto_optimizer(cls, model, lr=lr, total_steps=total_steps, **overrides)

    # ------------------------------------------------------------------ #
    # Checkpointing (schema v2 with v0.3-monolith migration)
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer state tagged with the PsiLogic schema version."""
        sd = super().state_dict()
        sd[_STATE_DICT_SCHEMA_KEY] = _STATE_DICT_SCHEMA_VERSION
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state, migrating pre-v0.4 (schema v1) checkpoints.

        v1 checkpoints (the v0.3 monolith) lack newer per-group keys such as
        ``gamma_auto``; those are filled from the current defaults. Missing
        chaos-state tensors are re-initialized to a neutral value — the EMAs
        renormalize within a handful of steps and shrinkage stays bounded by
        ``max_cancel`` throughout.
        """
        state_dict = dict(state_dict)
        schema = state_dict.pop(_STATE_DICT_SCHEMA_KEY, 1)
        if schema > _STATE_DICT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint uses PsiLogic state_dict schema v{schema}, but this "
                f"version only supports up to v{_STATE_DICT_SCHEMA_VERSION}. "
                "Please upgrade the psilogic package."
            )
        if schema < _STATE_DICT_SCHEMA_VERSION:
            state_dict = self._migrate_state_dict(state_dict)
        super().load_state_dict(state_dict)
        self._fill_missing_state()

    def _migrate_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        migrated = {
            "state": {key: dict(value) for key, value in state_dict["state"].items()},
            "param_groups": [dict(group) for group in state_dict["param_groups"]],
        }
        for group in migrated["param_groups"]:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
        return migrated

    def _fill_missing_state(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state.get(param)
                if not state:
                    continue
                if "gn_avg" not in state:
                    state["gn_avg"] = torch.zeros(1, device=param.device, dtype=torch.float32)
                for key in ("fast", "slow"):
                    if key not in state:
                        state[key] = torch.ones(1, device=param.device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # DDP chaos synchronization
    # ------------------------------------------------------------------ #

    def _maybe_sync_chaos_batched(self, fast_vec: torch.Tensor, slow_vec: torch.Tensor) -> None:
        """All-reduce (mean) stacked fast/slow chaos EMAs across DDP ranks.

        Equivalent to :meth:`_maybe_sync_chaos` but takes the already-stacked
        per-group vectors instead of a list of per-parameter state dicts.
        Every rank builds the ``[fast_0..fast_n, slow_0..slow_n]`` layout in
        the same (deterministic) parameter order, so summing per-position
        across ranks gives the exact same reduced values as the original
        per-parameter ``cat``/``all_reduce``/``copy_`` sequence — only the
        in-tensor layout (block vs. interleaved) differs, which has no
        effect on an elementwise sum-then-divide reduction.
        """
        if not self._sync_chaos_ddp:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        n = fast_vec.numel()
        flat = torch.cat((fast_vec, slow_vec)).float()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(dist.get_world_size())
        fast_vec.copy_(flat[:n])
        slow_vec.copy_(flat[n:])

    def _maybe_sync_chaos(self, states: list[dict[str, Any]]) -> None:
        """All-reduce (mean) fast/slow chaos EMAs across DDP ranks in-place."""
        if not self._sync_chaos_ddp or not states:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        flat = torch.cat([torch.cat((state["fast"], state["slow"])).float() for state in states])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(dist.get_world_size())
        for i, state in enumerate(states):
            state["fast"].copy_(flat[2 * i])
            state["slow"].copy_(flat[2 * i + 1])

    # ------------------------------------------------------------------ #
    # Update math
    # ------------------------------------------------------------------ #

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
        chaos_gain: float,
    ) -> None:
        if chaos_gain > 0.0 and gamma_eff > 0:
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
            total_scalar_decay = lr * wd + chaos_contrib * chaos_gain
            param.mul_(1.0 - total_scalar_decay)

            if qd_eff > 0:
                qd_contrib = qd_eff * chaos_gain * (1.0 - spike_mask)
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
            state["m"].mul_(beta2).add_(grad, alpha=1.0 - beta2)
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
        warmup_cfg = group["chaos_warmup"]
        adapt_tau = group["adaptive_tau"]
        tau_scale = group["tau_scale"]
        max_cancel = group["max_cancel"]
        agc = group["agc_clip"]
        gamma_t_max = group["gamma_T_max"]
        lion = group["lion_mode"]
        gamma_auto_on = group["gamma_auto"]

        # Phase 1 (prepare): grad processing, momentum update, and the
        # gradient-norm EMA update for every parameter in the group.
        prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], int]] = []
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

            if not lion:
                state["m"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
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

            prepared.append((param, raw_grad, grad, state, step))

        if not prepared:
            return

        # Phase 2 (sync): one batched all-reduce for the whole group instead
        # of one all-reduce per parameter.
        self._maybe_sync_chaos([state for (_, _, _, state, _) in prepared])

        # Phase 3 (finalize): chaos-conditioned decay and the Adam/Lion update.
        for param, raw_grad, grad, state, step in prepared:
            gamma_eff, qd_eff = effective_gamma_and_qd(step, gamma_t_max, gamma, qd)
            if gamma_auto_on:
                gamma_eff = float(auto_gamma(state["slow"], step, gamma_eff))
            chaos_gain = effective_warmup(step, gamma_t_max, warmup_cfg)

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
                chaos_gain=chaos_gain,
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

        grads = [p.grad for p in params_with_grad]

        if agc > 0.0:
            p_norms = torch.stack(torch._foreach_norm(params_with_grad))
            g_norms = torch.stack(torch._foreach_norm(grads))
            max_norms = agc * p_norms.clamp(min=1e-3)
            clip_factors = (max_norms / g_norms.clamp(min=1e-6)).clamp(max=1.0)
            torch._foreach_mul_(grads, clip_factors.unbind())

        raw_grads = list(grads)

        if gc:
            grads = [_centralize_grad(g) if g.dim() > 1 else g for g in grads]

        states = []
        for param in params_with_grad:
            state = self.state[param]
            if not state:
                _init_param_state(state, param)
            state["t"] += 1
            states.append(state)

        ms = [s["m"] for s in states]
        vs = [s["v"] for s in states]

        if not lion:
            torch._foreach_mul_(ms, beta1)
            torch._foreach_add_(ms, grads, alpha=1.0 - beta1)
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, grads, grads, value=1.0 - beta2)

        # Always recompute the norm directly from the final (post-AGC,
        # post-centralize) gradient tensor, exactly like `_step_scalar`'s
        # `grad.norm()`. The previous `g_norms * clip_factors` shortcut here
        # (valid only in exact arithmetic: ||c*x|| == c*||x||) saved one
        # `_foreach_norm` call but rounds differently from `(c*x).norm()` in
        # low precision, which desynced the fast/slow chaos EMA from the
        # scalar reference path under bf16/fp16.
        g_norms_vec = torch.stack(torch._foreach_norm(grads))

        uniform_step = len({state["t"] for state in states}) == 1
        homogeneous = len({(p.device, p.dtype) for p in params_with_grad}) == 1
        batched = uniform_step and homogeneous

        decay_scales: list[torch.Tensor] = []
        qd_factors: list[torch.Tensor] = []
        has_chaos_or_wd = False
        has_qd = False

        if batched:
            _group_step = states[0]["t"]
            _group_gamma_eff, _group_qd_eff = effective_gamma_and_qd(
                _group_step, gamma_t_max, gamma, qd
            )
            _group_chaos_gain = effective_warmup(_group_step, gamma_t_max, warmup_cfg)

            fast_vec = torch.cat([s["fast"] for s in states])
            slow_vec = torch.cat([s["slow"] for s in states])
            gn_avg_vec = torch.cat([s["gn_avg"] for s in states])

            sqrt_numels = torch.tensor(
                [math.sqrt(max(g.numel(), 1)) for g in grads],
                device=fast_vec.device,
                dtype=fast_vec.dtype,
            )
            gn_scaled_vec = g_norms_vec / sqrt_numels

            update_gradient_norm_ema_batched(
                gn_scaled_vec, _group_step, fast_vec, slow_vec, gn_avg_vec, eps
            )
            self._maybe_sync_chaos_batched(fast_vec, slow_vec)

            if _FOREACH_COPY_AVAILABLE:
                fast_slices = [fast_vec[i : i + 1] for i in range(len(states))]
                slow_slices = [slow_vec[i : i + 1] for i in range(len(states))]
                gn_avg_slices = [gn_avg_vec[i : i + 1] for i in range(len(states))]
                torch._foreach_copy_([s["fast"] for s in states], fast_slices)
                torch._foreach_copy_([s["slow"] for s in states], slow_slices)
                torch._foreach_copy_([s["gn_avg"] for s in states], gn_avg_slices)
            else:
                for i, state in enumerate(states):
                    state["fast"].copy_(fast_vec[i : i + 1])
                    state["slow"].copy_(slow_vec[i : i + 1])
                    state["gn_avg"].copy_(gn_avg_vec[i : i + 1])

            if _group_chaos_gain > 0.0 and _group_gamma_eff > 0:
                has_chaos_or_wd = True
                gamma_eff_vec: torch.Tensor | float
                if gamma_auto_on:
                    gamma_eff_vec = auto_gamma_batched(slow_vec, _group_step, _group_gamma_eff)
                else:
                    gamma_eff_vec = _group_gamma_eff

                chaos_contrib_vec, spike_mask_vec = chaos_contribution_batched(
                    slow_vec,
                    fast_vec,
                    adaptive_tau=adapt_tau,
                    chaos_tau=group["chaos_tau"],
                    tau_scale=tau_scale,
                    eps=eps,
                    lr=lr,
                    gamma_eff=gamma_eff_vec,
                    p_ext=p_ext,
                    max_cancel=max_cancel,
                    param_dtype=fast_vec.dtype,
                )
                decay_scale_vec = 1.0 - (lr * wd + chaos_contrib_vec * _group_chaos_gain)
                decay_scales = [decay_scale_vec[i : i + 1] for i in range(len(states))]

                if _group_qd_eff > 0:
                    has_qd = True
                    qd_contrib_vec = _group_qd_eff * _group_chaos_gain * (1.0 - spike_mask_vec)
                    for i, raw_grad in enumerate(raw_grads):
                        qd_factors.append(
                            1.0 - lr * qd_contrib_vec[i : i + 1] * torch.tanh(raw_grad.abs())
                        )
                else:
                    one = _one_scalar(fast_vec.device, fast_vec.dtype)
                    qd_factors = [one] * len(states)
            else:
                wd_only_scale = torch.tensor(
                    1.0 - lr * wd, device=fast_vec.device, dtype=fast_vec.dtype
                )
                decay_scales = [wd_only_scale] * len(states)
                if wd > 0:
                    has_chaos_or_wd = True
                one = _one_scalar(fast_vec.device, fast_vec.dtype)
                qd_factors = [one] * len(states)
        else:
            for gn, state, grad in zip(g_norms_vec.unbind(), states, grads):
                update_gradient_norm_ema(
                    gn, grad.numel(), state["t"], state["fast"], state["slow"], state["gn_avg"], eps
                )
            self._maybe_sync_chaos(states)

            wd_only_scale_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

            for param, raw_grad, state in zip(params_with_grad, raw_grads, states):
                p_step = state["t"]
                p_gamma_eff, p_qd_eff = effective_gamma_and_qd(p_step, gamma_t_max, gamma, qd)
                if gamma_auto_on:
                    p_gamma_eff = auto_gamma(state["slow"], p_step, p_gamma_eff)
                p_chaos_gain = effective_warmup(p_step, gamma_t_max, warmup_cfg)

                if p_chaos_gain > 0.0 and p_gamma_eff > 0:
                    chaos_contrib, spike_mask = chaos_contribution(
                        state["slow"],
                        state["fast"],
                        adaptive_tau=adapt_tau,
                        chaos_tau=group["chaos_tau"],
                        tau_scale=tau_scale,
                        eps=eps,
                        lr=lr,
                        gamma_eff=p_gamma_eff,
                        p_ext=p_ext,
                        max_cancel=max_cancel,
                        param_dtype=param.dtype,
                    )
                    total_scalar_decay = lr * wd + chaos_contrib * p_chaos_gain
                    decay_scales.append(1.0 - total_scalar_decay)
                    has_chaos_or_wd = True

                    if p_qd_eff > 0:
                        qd_contrib = p_qd_eff * p_chaos_gain * (1.0 - spike_mask)
                        qd_factors.append(1.0 - lr * qd_contrib * torch.tanh(raw_grad.abs()))
                        has_qd = True
                    else:
                        qd_factors.append(_one_scalar(param.device, param.dtype))
                else:
                    key = (param.device, param.dtype)
                    assert param.grad is not None
                    cached_scale = wd_only_scale_cache.get(key)
                    if cached_scale is None:
                        cached_scale = torch.tensor(
                            1.0 - lr * wd, device=param.device, dtype=param.dtype
                        )
                        wd_only_scale_cache[key] = cached_scale
                    decay_scales.append(cached_scale)
                    if wd > 0:
                        has_chaos_or_wd = True
                    qd_factors.append(_one_scalar(param.device, param.dtype))

        if has_chaos_or_wd:
            torch._foreach_mul_(params_with_grad, decay_scales)
        if has_qd:
            torch._foreach_mul_(params_with_grad, qd_factors)

        if lion:
            for param, momentum, grad in zip(params_with_grad, ms, grads):
                update = (beta1 * momentum + (1.0 - beta1) * grad).sign()
                param.add_(update, alpha=-lr)
                momentum.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        else:
            if lr > 0.0:
                bc1_list = [1.0 - beta1 ** state["t"] for state in states]
                bc2_sqrt_list = [math.sqrt(1.0 - beta2 ** state["t"]) for state in states]
                step_sizes_neg = [
                    -lr * bc2_sqrt / bc1 for bc1, bc2_sqrt in zip(bc1_list, bc2_sqrt_list)
                ]

                denoms = torch._foreach_sqrt(vs)
                torch._foreach_add_(denoms, eps)
                if uniform_step:
                    torch._foreach_addcdiv_(params_with_grad, ms, denoms, value=step_sizes_neg[0])
                else:
                    for param, momentum, denom, step_size in zip(
                        params_with_grad, ms, denoms, step_sizes_neg
                    ):
                        param.addcdiv_(momentum, denom, value=step_size)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        t_start = time.perf_counter() if self._profile_step_time else 0.0

        is_compiling = getattr(torch.compiler, "is_compiling", lambda: False)()

        for group in self.param_groups:
            has_cuda_grad = any(p.is_cuda for p in group["params"] if p.grad is not None)
            use_fused = (
                self._use_fused_cuda
                and group.get("use_fused_cuda", self._use_fused_cuda)
                and has_cuda_grad
                and not is_compiling
            )
            use_foreach = group["use_foreach"] and _FOREACH_AVAILABLE and has_cuda_grad
            if use_fused:
                self._step_fused_cuda(group)
            elif use_foreach:
                self._step_foreach(group)
            else:
                self._step_scalar(group)

        if self._profile_step_time:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.last_step_time_ms = elapsed_ms
            if self.step_time_ms_ema is None:
                self.step_time_ms_ema = elapsed_ms
            else:
                self.step_time_ms_ema = 0.9 * self.step_time_ms_ema + 0.1 * elapsed_ms

        return loss

    def _step_fused_cuda(self, group: dict[str, Any]) -> None:
        from ._cuda import fused_group_step, is_fused_available

        if not is_fused_available():
            self._step_foreach(group)
            return
        fused_group_step(
            group,
            self.state,
            sync_chaos_ddp=self._sync_chaos_ddp,
            maybe_sync=self._maybe_sync_chaos,
        )
