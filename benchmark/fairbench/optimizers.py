"""Optimizer construction and a clean reference Lion implementation.

The benchmark compares four optimizers behind one uniform factory
(:func:`build_optimizer`) so the training loop never special-cases an
optimizer. Two design rules keep the comparison fair:

* Identical *coupling* of weight decay where the algorithm allows it. Adam
  uses classic L2 (its canonical form), while AdamW, Lion and PsiLogic use
  decoupled decay (their canonical form). We do **not** retro-fit one
  algorithm's regularizer onto another -- each runs as published.
* Identical performance switches (``foreach`` / fused batched CUDA ops) are
  enabled wherever the optimizer supports them.

Lion is imported from ``lion_pytorch`` or ``pytorch_optimizer`` when present,
otherwise the self-contained reference implementation below is used so the
benchmark has zero hard third-party optimizer dependency.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

# --------------------------------------------------------------------------- #
# Reference Lion (Chen et al., 2023, "Symbolic Discovery of Optimization
# Algorithms"). Sign-momentum update with decoupled weight decay.
# --------------------------------------------------------------------------- #


class Lion(Optimizer):
    r"""EvoLved Sign Momentum optimizer.

    Update rule (per parameter)::

        c_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
        theta = theta - lr * (sign(c_t) + wd * theta)      # decoupled decay
        m_t   = beta2 * m_{t-1} + (1 - beta2) * g_t

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate. Lion's effective step is ~3-10x smaller than Adam's,
            so good LRs are typically ~1e-4.
        betas: ``(beta1, beta2)`` for the interpolation / momentum EMAs.
        weight_decay: Decoupled (AdamW-style) weight decay.
        use_foreach: Use batched ``torch._foreach_*`` ops on CUDA tensors.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_foreach: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, use_foreach=use_foreach)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state["exp_avg"])

            if not params:
                continue

            foreach = group["use_foreach"] and len(params) > 1 and all(p.is_cuda for p in params)
            if foreach:
                self._step_foreach(params, grads, exp_avgs, lr, beta1, beta2, wd)
            else:
                self._step_scalar(params, grads, exp_avgs, lr, beta1, beta2, wd)

        return loss

    @staticmethod
    def _step_scalar(params, grads, exp_avgs, lr, beta1, beta2, wd) -> None:
        for p, grad, exp_avg in zip(params, grads, exp_avgs):
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
            p.add_(update, alpha=-lr)
            exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

    @staticmethod
    def _step_foreach(params, grads, exp_avgs, lr, beta1, beta2, wd) -> None:
        if wd != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * wd)
        # update = sign(beta1 * exp_avg + (1 - beta1) * grad)
        updates = torch._foreach_mul(exp_avgs, beta1)
        torch._foreach_add_(updates, grads, alpha=1.0 - beta1)
        updates = torch._foreach_sign(updates)
        torch._foreach_add_(params, updates, alpha=-lr)
        # exp_avg = beta2 * exp_avg + (1 - beta2) * grad
        torch._foreach_mul_(exp_avgs, beta2)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta2)


def _import_external_lion():
    """Return an external Lion class if a known package is installed, else None."""
    try:
        from lion_pytorch import Lion as _Lion  # type: ignore

        return _Lion
    except Exception:
        pass
    try:
        from pytorch_optimizer import Lion as _Lion  # type: ignore

        return _Lion
    except Exception:
        pass
    return None


def _import_psilogic():
    """Return the PsiLogic optimizer class, raising a helpful error if absent."""
    try:
        from psilogic import PsiLogic  # type: ignore

        return PsiLogic
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The 'psilogic' package is required for the PsiLogic optimizer. "
            "Install it with `pip install psilogic`."
        ) from exc


# Default hyperparameters held constant across the sweep/training so that the
# *only* tuned axis is the learning rate (the Fair-Play protocol). These are
# each optimizer's canonical, widely-used defaults.
DEFAULT_BETAS_ADAM: tuple[float, float] = (0.9, 0.999)
DEFAULT_BETAS_LION: tuple[float, float] = (0.9, 0.99)
DEFAULT_EPS: float = 1e-8


def build_optimizer(
    name: str,
    params: Iterable[Tensor],
    lr: float,
    weight_decay: float = 0.0,
    use_foreach: bool = True,
    psilogic_kwargs: Optional[dict[str, Any]] = None,
) -> Optimizer:
    """Construct an optimizer by canonical name.

    Args:
        name: One of ``{"adam", "adamw", "lion", "psilogic"}`` (case-insensitive).
        params: Parameters or parameter groups to optimize.
        lr: Learning rate (the only axis tuned by the Fair-Play sweep).
        weight_decay: Weight decay. Coupled (L2) for Adam, decoupled otherwise.
        use_foreach: Enable batched CUDA kernels where supported.
        psilogic_kwargs: Extra keyword args forwarded to ``PsiLogic`` (e.g.
            per-arena ``gamma``); ``lr``/``weight_decay``/``use_foreach`` are
            always taken from the explicit arguments.

    Returns:
        A configured :class:`torch.optim.Optimizer`.
    """
    key = name.lower()
    params = list(params)

    if key == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=DEFAULT_BETAS_ADAM,
            eps=DEFAULT_EPS,
            weight_decay=weight_decay,  # classic coupled L2 -- Adam's canonical form
            foreach=use_foreach or None,
        )

    if key == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=DEFAULT_BETAS_ADAM,
            eps=DEFAULT_EPS,
            weight_decay=weight_decay,  # decoupled -- AdamW's canonical form
            foreach=use_foreach or None,
        )

    if key == "lion":
        external = _import_external_lion()
        if external is not None:
            try:
                return external(
                    params,
                    lr=lr,
                    betas=DEFAULT_BETAS_LION,
                    weight_decay=weight_decay,
                    use_triton=False,
                )
            except TypeError:
                # Some forks lack the use_triton kwarg.
                return external(params, lr=lr, betas=DEFAULT_BETAS_LION, weight_decay=weight_decay)
        return Lion(
            params,
            lr=lr,
            betas=DEFAULT_BETAS_LION,
            weight_decay=weight_decay,
            use_foreach=use_foreach,
        )

    if key == "psilogic":
        PsiLogic = _import_psilogic()
        kwargs: dict[str, Any] = dict(psilogic_kwargs or {})
        kwargs.update(lr=lr, weight_decay=weight_decay, use_foreach=use_foreach)
        return PsiLogic(params, **kwargs)

    raise ValueError(
        f"Unknown optimizer '{name}'. Expected one of {('adam', 'adamw', 'lion', 'psilogic')}."
    )


def is_psilogic(optimizer: Optimizer) -> bool:
    """True if ``optimizer`` is a PsiLogic instance (probed without importing)."""
    return type(optimizer).__name__ == "PsiLogic" or any(
        base.__name__ == "PsiLogic" for base in type(optimizer).__mro__
    )
