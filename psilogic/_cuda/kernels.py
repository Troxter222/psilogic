"""Triton kernels for the fused PsiLogic CUDA step."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _require_triton() -> None:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not installed")


if _HAS_TRITON:

    @triton.jit
    def _leader_sum_kernel(
        g_ptr,
        leader_sum_ptr,
        n_elements,
        elems_per_leader,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        leader = offsets // elems_per_leader
        tl.atomic_add(leader_sum_ptr + leader, g, mask=mask)

    @triton.jit
    def _centralize_moment_kernel(
        g_ptr,
        raw_g_ptr,
        m_ptr,
        v_ptr,
        leader_sum_ptr,
        leader_count,
        beta1,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        grad_centralize,
        update_variance,
        n_elements,
        elems_per_leader,
        n_leaders,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        tl.store(raw_g_ptr + offsets, g, mask=mask)

        if grad_centralize:
            leader = offsets // elems_per_leader
            lsum = tl.load(leader_sum_ptr + leader, mask=leader < n_leaders, other=0.0)
            g = g - lsum / leader_count

        new_m = beta1 * m + one_minus_beta1 * g
        tl.store(m_ptr + offsets, new_m, mask=mask)
        tl.store(g_ptr + offsets, g, mask=mask)

        if update_variance:
            v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            new_v = beta2 * v + one_minus_beta2 * g * g
            tl.store(v_ptr + offsets, new_v, mask=mask)

    @triton.jit
    def _decay_adam_kernel(
        g_ptr,
        raw_g_ptr,
        p_ptr,
        m_ptr,
        v_ptr,
        lr,
        eps,
        step_size,
        total_scalar_decay,
        wd_only_decay,
        qd_contrib,
        apply_quantum,
        lion,
        beta1,
        one_minus_beta1,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if total_scalar_decay > 0.0:
            p = p * (1.0 - total_scalar_decay)
        elif wd_only_decay > 0.0:
            p = p * (1.0 - wd_only_decay)

        if apply_quantum:
            raw_g = tl.load(raw_g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            p = p * (1.0 - lr * qd_contrib * tl.tanh(tl.abs(raw_g)))

        if lion:
            g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            update = beta1 * m + one_minus_beta1 * g
            sign = tl.where(update > 0, 1.0, tl.where(update < 0, -1.0, 0.0))
            p = p - lr * sign
        else:
            v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            denom = tl.sqrt(v) + eps
            p = p - step_size * m / denom

        tl.store(p_ptr + offsets, p, mask=mask)


def launch_leader_sums(grad: torch.Tensor, n_leaders: int, elems_per_leader: int) -> torch.Tensor:
    _require_triton()
    leader_sum = torch.zeros(n_leaders, device=grad.device, dtype=torch.float32)
    n = grad.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _leader_sum_kernel[grid](grad, leader_sum, n, elems_per_leader, BLOCK=block)
    return leader_sum


def launch_centralize_moment(
    grad: torch.Tensor,
    raw_grad: torch.Tensor,
    momentum: torch.Tensor,
    variance: torch.Tensor,
    *,
    grad_centralize: bool,
    beta1: float,
    beta2: float,
    update_variance: bool,
    n_leaders: int,
    elems_per_leader: int,
    leader_sum: torch.Tensor,
) -> None:
    """Centralize grad (optional), snapshot raw grad, update Adam moments."""
    _require_triton()
    n = grad.numel()
    if n == 0:
        return
    block = 256
    grid = (triton.cdiv(n, block),)
    _centralize_moment_kernel[grid](
        grad,
        raw_grad,
        momentum,
        variance,
        leader_sum,
        float(elems_per_leader),
        float(beta1),
        float(beta2),
        float(1.0 - beta1),
        float(1.0 - beta2),
        grad_centralize and n_leaders > 1 and elems_per_leader > 1,
        update_variance,
        n,
        elems_per_leader,
        n_leaders,
        BLOCK=block,
    )


def launch_decay_adam(
    grad: torch.Tensor,
    raw_grad: torch.Tensor,
    param: torch.Tensor,
    momentum: torch.Tensor,
    variance: torch.Tensor,
    *,
    lr: float,
    eps: float,
    step_size: float,
    total_scalar_decay: float,
    wd_only_decay: float,
    qd_contrib: float,
    apply_quantum: bool,
    lion: bool,
    beta1: float,
) -> None:
    """Apply chaos/weight decay and Adam or Lion param update."""
    _require_triton()
    n = param.numel()
    if n == 0:
        return
    block = 256
    grid = (triton.cdiv(n, block),)
    _decay_adam_kernel[grid](
        grad,
        raw_grad,
        param,
        momentum,
        variance,
        float(lr),
        float(eps),
        float(step_size),
        float(total_scalar_decay),
        float(wd_only_decay),
        float(qd_contrib),
        apply_quantum,
        lion,
        float(beta1),
        float(1.0 - beta1),
        n,
        BLOCK=block,
    )
