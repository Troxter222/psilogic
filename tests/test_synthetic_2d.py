"""2D synthetic loss convergence tests for PsiLogic."""

from __future__ import annotations

import torch
import torch.nn as nn

from psilogic import PsiLogic


def optimize_2d(
    loss_fn,
    init: tuple[float, float],
    lr: float,
    n_steps: int,
    **psi_kwargs,
) -> float:
    """Minimize a 2D loss with PsiLogic; return the final loss value."""
    torch.manual_seed(0)
    x = nn.Parameter(torch.tensor([init[0]], dtype=torch.float32))
    y = nn.Parameter(torch.tensor([init[1]], dtype=torch.float32))
    opt = PsiLogic([x, y], lr=lr, weight_decay=0.0, **psi_kwargs)

    for _ in range(n_steps):
        opt.zero_grad()
        loss = loss_fn(x.squeeze(), y.squeeze())
        loss.backward()
        opt.step()

    with torch.no_grad():
        return loss_fn(x.squeeze(), y.squeeze()).item()


def _sphere_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * x + y * y


def _rosenbrock_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (1 - x).pow(2) + 100 * (y - x.pow(2)).pow(2)


def test_sphere_convergence() -> None:
    final = optimize_2d(_sphere_loss, init=(3.0, 4.0), lr=0.1, n_steps=100, gamma=0.05)
    assert final < 1e-3, f"Sphere loss did not converge: {final:.6f}"


def test_rosenbrock_convergence() -> None:
    init = (-1.2, 1.0)
    init_loss = _rosenbrock_loss(
        torch.tensor(init[0], dtype=torch.float32),
        torch.tensor(init[1], dtype=torch.float32),
    ).item()
    final = optimize_2d(_rosenbrock_loss, init=init, lr=1e-2, n_steps=1500, gamma=0.05)
    assert final < init_loss, f"Rosenbrock loss did not decrease: {init_loss:.4f} → {final:.4f}"
    assert final < 0.5, f"Rosenbrock loss too high: {final:.4f}"
