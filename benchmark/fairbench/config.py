"""Typed configuration objects for the FairBench benchmark.

All experiment knobs live here as ``dataclass`` instances so that a run is
fully described by a single serializable object. This keeps experiments
reproducible (the config is logged verbatim to CSV / W&B) and keeps the
rest of the codebase free of scattered magic numbers.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Canonical names. Using constants avoids silent typos across the codebase.
# --------------------------------------------------------------------------- #

OPTIMIZERS: tuple[str, ...] = ("adam", "adamw", "lion", "psilogic")
ARENAS: tuple[str, ...] = ("nlp", "vit", "resnet", "diffusion")

#: The optimizer treated as the "method under test" for significance testing.
REFERENCE_OPTIMIZER: str = "psilogic"


@dataclass
class SweepConfig:
    """Stage-1 learning-rate sweep settings.

    The sweep runs each optimizer for a short budget and selects the LR that
    minimizes validation loss. This is what removes per-optimizer *tuning
    bias*: every optimizer is given the same opportunity to find its best LR.
    """

    #: Log-spaced grid of candidate learning rates (inclusive endpoints).
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    num_lrs: int = 7
    #: Budget for each trial. Whichever limit is hit first stops the trial.
    max_steps: int = 500
    max_epochs: int = 5
    #: If True, an explicit grid overrides the (lr_min, lr_max, num_lrs) range.
    explicit_grid: Optional[list[float]] = None

    def grid(self) -> list[float]:
        """Return the concrete list of learning rates to evaluate."""
        if self.explicit_grid:
            return sorted(set(self.explicit_grid))
        # Geometric (log-uniform) spacing -- the standard choice for LR search.
        import numpy as np

        lrs = np.geomspace(self.lr_min, self.lr_max, self.num_lrs)
        return [float(x) for x in lrs]


@dataclass
class TrainConfig:
    """Stage-2 full-training settings, shared across arenas."""

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    max_steps: int = 2000
    max_epochs: int = 10
    #: Steps between validation passes. ``0`` -> validate once per epoch only.
    eval_every: int = 200
    #: Gradient clipping (global norm). ``0`` disables.
    grad_clip: float = 1.0
    #: Optional linear warmup (in steps) followed by cosine decay to 0.
    warmup_steps: int = 100
    use_scheduler: bool = True


@dataclass
class HardwareConfig:
    """Single-GPU hardware / performance knobs."""

    device: str = "cuda"
    #: Automatic mixed precision via ``torch.amp.autocast``.
    amp: bool = True
    amp_dtype: str = "float16"  # one of {"float16", "bfloat16"}
    #: Pass ``foreach=True`` / ``use_foreach=True`` to optimizers when possible.
    use_foreach: bool = True
    #: ``torch.compile`` the model (PyTorch >= 2.0). Off by default for fairness
    #: and portability; the optimizer is what we benchmark, not the compiler.
    compile_model: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    #: cudnn.benchmark -- safe speed-up for fixed input sizes.
    cudnn_benchmark: bool = True


@dataclass
class ArenaConfig:
    """Per-arena data / model / batching configuration.

    Defaults are intentionally modest so the full benchmark is tractable on a
    single consumer GPU. Override via the CLI or a JSON config for full runs.
    """

    name: str
    batch_size: int = 64
    data_root: str = "./data"
    #: Arena-specific extra options (model width, image size, etc.).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Where and how results are recorded."""

    output_dir: str = "./results"
    run_name: Optional[str] = None
    csv: bool = True
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "fairbench-optimizers"
    wandb_entity: Optional[str] = None
    #: Make learning-curve plots (PNG) with ±std shaded confidence bands.
    plots: bool = True
    #: Log scalars at most every ``log_every`` steps to keep overhead low.
    log_every: int = 10


@dataclass
class BenchmarkConfig:
    """Top-level configuration describing an entire benchmark invocation."""

    arenas: list[str] = field(default_factory=lambda: list(ARENAS))
    optimizers: list[str] = field(default_factory=lambda: list(OPTIMIZERS))
    sweep: SweepConfig = field(default_factory=SweepConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    arena_configs: dict[str, ArenaConfig] = field(default_factory=dict)

    #: Skip Stage 1 and use these LRs directly: ``{arena: {optimizer: lr}}``.
    fixed_lrs: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Synthetic-data mode: tiny generated datasets for CI / smoke tests.
    synthetic: bool = False
    #: Never download; require pre-staged datasets under ``data_root``.
    offline: bool = False
    seed_base: int = 1234

    def arena_config(self, arena: str) -> ArenaConfig:
        """Return (creating if needed) the :class:`ArenaConfig` for ``arena``."""
        if arena not in self.arena_configs:
            self.arena_configs[arena] = ArenaConfig(name=arena)
        return self.arena_configs[arena]

    # ----------------------------- (de)serialization -------------------- #

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Rebuild a config from a (possibly partial) dictionary."""
        data = dict(data)
        sweep = SweepConfig(**data.pop("sweep", {}) or {})
        train = TrainConfig(**data.pop("train", {}) or {})
        hardware = HardwareConfig(**data.pop("hardware", {}) or {})
        logging_cfg = LoggingConfig(**data.pop("logging", {}) or {})
        arena_cfgs_raw = data.pop("arena_configs", {}) or {}
        arena_cfgs = {k: ArenaConfig(**v) for k, v in arena_cfgs_raw.items()}
        return cls(
            sweep=sweep,
            train=train,
            hardware=hardware,
            logging=logging_cfg,
            arena_configs=arena_cfgs,
            **data,
        )

    @classmethod
    def from_json(cls, path: str) -> BenchmarkConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
