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
        raw_input_ptr,
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
        update_momentum,
        n_elements,
        elems_per_leader,
        n_leaders,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        # Under eager dispatch (kernel[grid](...)) Triton infers these plain
        # Python floats as fp32. Under torch.compile/Inductor's generated
        # wrapper the same call site specializes them as fp64 doubles
        # instead (matching aten's default scalar promotion), which makes
        # ``g``'s dtype diverge between the two branches of the
        # ``if grad_centralize`` block below (fp32 on the implicit "else",
        # fp64 on the "then") — Triton rejects that at compile time. Casting
        # to fp32 up front pins the dtype so both branches agree, with no
        # change to the arithmetic itself (all these scalars are already
        # meant to be single precision here).
        leader_count = leader_count.to(tl.float32)
        beta1 = beta1.to(tl.float32)
        beta2 = beta2.to(tl.float32)
        one_minus_beta1 = one_minus_beta1.to(tl.float32)
        one_minus_beta2 = one_minus_beta2.to(tl.float32)

        raw_g = tl.load(raw_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        tl.store(raw_g_ptr + offsets, raw_g, mask=mask)

        if grad_centralize:
            leader = offsets // elems_per_leader
            lsum = tl.load(leader_sum_ptr + leader, mask=leader < n_leaders, other=0.0)
            g = g - lsum / leader_count

        new_m = beta1 * m + one_minus_beta1 * g
        if update_momentum:
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
        total_scalar_decay_ptr,  # was: total_scalar_decay (float) — now a 1-elem fp32 pointer
        wd_only_decay_ptr,  # was: wd_only_decay (float)      — now a 1-elem fp32 pointer
        qd_contrib_ptr,  # was: qd_contrib (float)         — now a 1-elem fp32 pointer
        apply_quantum,
        lion,
        beta1,
        one_minus_beta1,
        beta2,
        one_minus_beta2,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        # Same fp32/fp64 specialization hazard as `_centralize_moment_kernel`
        # above: under eager dispatch these Python-float kernel arguments
        # infer as fp32, but torch.compile/Inductor's wrapper can specialize
        # the same call site to fp64, which then makes `p`/`m` diverge in
        # dtype between the `if`/`else` branches below and Triton rejects
        # it at compile time. Pin them to fp32 up front — no change to the
        # arithmetic, these were always meant to be single precision here.
        lr = lr.to(tl.float32)
        eps = eps.to(tl.float32)
        step_size = step_size.to(tl.float32)
        beta1 = beta1.to(tl.float32)
        one_minus_beta1 = one_minus_beta1.to(tl.float32)
        beta2 = beta2.to(tl.float32)
        one_minus_beta2 = one_minus_beta2.to(tl.float32)

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Scalars now live on-device; read them here instead of baking them
        # in as Python-float kernel arguments computed via a blocking `.item()`
        # on the host. Same values, same math — just no CPU/GPU sync to get them.
        total_scalar_decay = tl.load(total_scalar_decay_ptr).to(tl.float32)
        wd_only_decay = tl.load(wd_only_decay_ptr).to(tl.float32)

        if total_scalar_decay > 0.0:
            p = p * (1.0 - total_scalar_decay)
        elif wd_only_decay > 0.0:
            p = p * (1.0 - wd_only_decay)

        if apply_quantum:
            raw_g = tl.load(raw_g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            qd_contrib = tl.load(qd_contrib_ptr).to(tl.float32)
            # Triton does not expose ``tl.tanh`` in all supported versions.
            # For non-negative x, tanh(x) == 2 * sigmoid(2x) - 1.
            p = p * (1.0 - lr * qd_contrib * (2.0 * tl.sigmoid(2.0 * tl.abs(raw_g)) - 1.0))

        if lion:
            g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            update = beta1 * m + one_minus_beta1 * g
            sign = tl.where(update > 0, 1.0, tl.where(update < 0, -1.0, 0.0))
            p = p - lr * sign
            m = beta2 * m + one_minus_beta2 * g
        else:
            v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            denom = tl.sqrt(v) + eps
            p = p + (-step_size) * (m / denom)

        tl.store(p_ptr + offsets, p, mask=mask)
        if lion:
            tl.store(m_ptr + offsets, m, mask=mask)


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
    raw_grad_source: torch.Tensor,
    raw_grad: torch.Tensor,
    momentum: torch.Tensor,
    variance: torch.Tensor,
    *,
    grad_centralize: bool,
    beta1: float,
    beta2: float,
    update_variance: bool,
    update_momentum: bool,
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
        raw_grad_source,
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
        update_momentum,
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
    total_scalar_decay: torch.Tensor,
    wd_only_decay: torch.Tensor,
    qd_contrib: torch.Tensor,
    apply_quantum: bool,
    lion: bool,
    beta1: float,
    beta2: float,
) -> None:
    """Apply chaos/weight decay and Adam or Lion param update.

    ``total_scalar_decay``, ``wd_only_decay``, and ``qd_contrib`` are now
    1-element float32 CUDA tensors (not Python floats). The kernel reads
    them straight off the device pointer, so the launch never forces a
    device-to-host sync — the values are exactly what ``.item()`` would
    have returned, just never pulled onto the CPU.
    """
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
        total_scalar_decay,
        wd_only_decay,
        qd_contrib,
        apply_quantum,
        lion,
        float(beta1),
        float(1.0 - beta1),
        float(beta2),
        float(1.0 - beta2),
        n,
        BLOCK=block,
    )
