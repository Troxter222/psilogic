"""Training engine, LR sweeper and the top-level benchmark orchestrator.

Layers, smallest to largest:

* :class:`TrainEngine` -- runs exactly one ``(optimizer, lr, seed)`` training
  job and returns its metrics/history. Task-agnostic; all task logic lives in
  the arena.
* :class:`LRSweeper` -- Stage 1 of the Fair-Play protocol: short per-optimizer
  LR search that removes tuning bias.
* :class:`BenchmarkRunner` -- Stage 2 + bookkeeping: identical-init multi-seed
  training, CSV/plot export and Mean+/-Std + t-test statistics.
"""

from __future__ import annotations

import copy
import gc
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .arenas import Arena, build_arena
from .config import REFERENCE_OPTIMIZER, BenchmarkConfig
from .logging_utils import LOGGER, ExperimentLogger, setup_console_logging
from .metrics import (
    Aggregate,
    CSVLogger,
    StepTimer,
    aggregate,
    cuda_sync,
    holm_bonferroni,
    paired_ttest,
    peak_vram_mb,
    reset_peak_vram,
)
from .optimizers import build_optimizer
from .probe import psilogic_chaos_metrics
from .utils import (
    amp_dtype_from_str,
    cosine_warmup_lambda,
    count_parameters,
    describe_device,
    format_device_label,
    is_oom_error,
    log_device_banner,
    make_autocast,
    resolve_device,
    save_config_with_hardware,
    set_seed,
)

# --------------------------------------------------------------------------- #
# Results containers
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    """Outcome of one training job."""

    arena: str
    optimizer: str
    seed: int
    lr: float
    final_metrics: dict[str, float] = field(default_factory=dict)
    primary_value: float = float("nan")
    step_history: list[dict[str, Any]] = field(default_factory=list)
    failed: bool = False
    error: Optional[str] = None


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    """Build a DataLoader whose shuffle order is reproducible from ``seed``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator if shuffle else None,
    )


# --------------------------------------------------------------------------- #
# Training engine
# --------------------------------------------------------------------------- #


class TrainEngine:
    """Runs a single training job for one ``(optimizer, lr, seed)`` combination."""

    def __init__(
        self,
        arena: Arena,
        cfg: BenchmarkConfig,
        device: torch.device,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device_info: Optional[dict[str, Any]] = None,
    ):
        self.arena = arena
        self.cfg = cfg
        self.device = device
        self.device_info = device_info or describe_device(device)
        self.gpu_label = format_device_label(self.device_info)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.amp_dtype = amp_dtype_from_str(cfg.hardware.amp_dtype)
        self.amp_ctx = make_autocast(device, cfg.hardware.amp, self.amp_dtype)
        # GradScaler is only needed for float16 on CUDA.
        self._use_scaler = (
            cfg.hardware.amp and device.type == "cuda" and self.amp_dtype == torch.float16
        )

    def _build_scaler(self):
        try:
            return torch.amp.GradScaler(self.device.type, enabled=self._use_scaler)
        except (TypeError, AttributeError):  # older torch
            return torch.cuda.amp.GradScaler(enabled=self._use_scaler)

    def run(
        self,
        optimizer_name: str,
        lr: float,
        seed: int,
        init_state: dict[str, torch.Tensor],
        max_steps: int,
        max_epochs: int,
        eval_every: int,
        logger: Optional[ExperimentLogger] = None,
        record_history: bool = True,
        use_scheduler: bool = True,
    ) -> RunResult:
        arena = self.arena
        cfg = self.cfg
        result = RunResult(arena.name, optimizer_name, seed, lr)

        # --- reproducible setup: identical model start across optimizers ----
        set_seed(seed)
        model = arena.build_model().to(self.device)
        model.load_state_dict(init_state)
        if cfg.hardware.compile_model and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)  # type: ignore
            except Exception as exc:
                LOGGER.warning("torch.compile failed (%s); continuing uncompiled.", exc)
        model.train()

        train_loader = _make_loader(
            self.train_dataset,
            arena.batch_size,
            True,
            seed,
            cfg.hardware.num_workers,
            cfg.hardware.pin_memory,
            drop_last=True,
        )
        val_loader = _make_loader(
            self.val_dataset,
            arena.batch_size,
            False,
            seed,
            cfg.hardware.num_workers,
            cfg.hardware.pin_memory,
            drop_last=False,
        )

        steps_per_epoch = len(train_loader)
        total_steps = min(max_steps, max_epochs * steps_per_epoch) if steps_per_epoch else max_steps

        optimizer = build_optimizer(
            optimizer_name,
            model.parameters(),
            lr=lr,
            weight_decay=arena.default_weight_decay,
            use_foreach=cfg.hardware.use_foreach,
            psilogic_kwargs=arena.psilogic_kwargs() if optimizer_name == "psilogic" else None,
        )
        scheduler = None
        if use_scheduler and cfg.train.use_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, cosine_warmup_lambda(cfg.train.warmup_steps, total_steps)
            )
        scaler = self._build_scaler()

        LOGGER.info(
            "[%s/%s/seed%d] lr=%.2e | %s | %s params=%.2fM | %d steps",
            arena.name,
            optimizer_name,
            seed,
            lr,
            self.gpu_label,
            type(optimizer).__name__,
            count_parameters(model) / 1e6,
            total_steps,
        )

        reset_peak_vram(self.device)
        global_step = 0
        running_peak_vram = 0.0
        t_run_start = time.perf_counter()

        try:
            for epoch in range(max_epochs):
                t_epoch = time.perf_counter()
                for batch in train_loader:
                    if global_step >= total_steps:
                        break

                    with StepTimer(self.device) as step_timer:
                        optimizer.zero_grad(set_to_none=True)
                        with self.amp_ctx():
                            loss, n_examples = arena.forward_loss(model, batch, self.device)
                        scaler.scale(loss).backward()
                        if cfg.train.grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        if scheduler is not None:
                            scheduler.step()

                    global_step += 1
                    loss_val = float(loss.detach())
                    if not math.isfinite(loss_val):
                        LOGGER.warning(
                            "[%s/%s/seed%d] non-finite loss at step %d; stopping run.",
                            arena.name,
                            optimizer_name,
                            seed,
                            global_step,
                        )
                        result.failed = True
                        result.error = "non-finite-loss"
                        raise StopIteration

                    running_peak_vram = max(running_peak_vram, peak_vram_mb(self.device))

                    # Throttled scalar logging / history recording.
                    if global_step % cfg.logging.log_every == 0 or global_step == 1:
                        throughput = n_examples / max(step_timer.elapsed, 1e-9)
                        step_metrics = {
                            "train_loss": loss_val,
                            "lr": optimizer.param_groups[0]["lr"],
                            "step_time_s": step_timer.elapsed,
                            "throughput_ex_s": throughput,
                            "peak_vram_mb": running_peak_vram,
                        }
                        step_metrics.update(psilogic_chaos_metrics(optimizer))
                        if logger is not None:
                            logger.log(step_metrics, global_step)
                        if record_history:
                            row = {"step": global_step, "epoch": epoch, "split": "train"}
                            row.update(step_metrics)
                            result.step_history.append(row)

                    # Periodic validation.
                    if eval_every and global_step % eval_every == 0:
                        val_metrics = self._evaluate(model, val_loader)
                        if logger is not None:
                            logger.log({f"val/{k}": v for k, v in val_metrics.items()}, global_step)
                        if record_history:
                            row = {"step": global_step, "epoch": epoch, "split": "val"}
                            row.update(val_metrics)
                            result.step_history.append(row)

                if global_step >= total_steps:
                    break
                # Per-epoch wall-clock logging.
                if logger is not None:
                    logger.log({"epoch_time_s": time.perf_counter() - t_epoch}, global_step)

        except StopIteration:
            pass  # divergence early-stop; metrics below reflect last good state
        except Exception as exc:  # includes CUDA OOM -- propagate for retry logic
            if is_oom_error(exc):
                raise
            LOGGER.exception(
                "[%s/%s/seed%d] unexpected error: %s", arena.name, optimizer_name, seed, exc
            )
            result.failed = True
            result.error = str(exc)

        # --- final evaluation ---------------------------------------------
        final_metrics = self._evaluate(model, val_loader)
        final_metrics["peak_vram_mb"] = running_peak_vram
        final_metrics["wall_time_s"] = time.perf_counter() - t_run_start
        result.final_metrics = final_metrics
        result.primary_value = final_metrics.get(arena.primary_metric, float("nan"))

        if logger is not None:
            logger.log({f"final/{k}": v for k, v in final_metrics.items()}, global_step)

        # Free GPU memory before the next job.
        del model, optimizer
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, val_loader: DataLoader) -> dict[str, float]:
        try:
            return self.arena.evaluate(model, val_loader, self.device, self.amp_ctx)
        except Exception as exc:
            if is_oom_error(exc):
                raise
            LOGGER.warning("Evaluation failed (%s).", exc)
            return {self.arena.primary_metric: float("nan")}


# --------------------------------------------------------------------------- #
# Stage 1: learning-rate sweep
# --------------------------------------------------------------------------- #


class LRSweeper:
    """Per-optimizer short LR search selecting the best validation metric."""

    def __init__(self, engine: TrainEngine, cfg: BenchmarkConfig):
        self.engine = engine
        self.cfg = cfg

    def sweep(
        self, optimizer_name: str, init_state: dict[str, torch.Tensor], seed: int
    ) -> tuple[float, list[dict[str, Any]]]:
        arena = self.engine.arena
        grid = self.cfg.sweep.grid()
        best_lr = grid[len(grid) // 2]  # geometric-median default
        best_value = -math.inf if arena.primary_mode == "max" else math.inf
        trials: list[dict[str, Any]] = []

        LOGGER.info(
            "[%s/%s] LR sweep over %s", arena.name, optimizer_name, [f"{x:.1e}" for x in grid]
        )
        for lr in grid:
            try:
                res = self.engine.run(
                    optimizer_name,
                    lr,
                    seed,
                    init_state,
                    max_steps=self.cfg.sweep.max_steps,
                    max_epochs=self.cfg.sweep.max_epochs,
                    eval_every=0,  # only final eval matters for selection
                    logger=None,
                    record_history=False,
                    use_scheduler=False,  # short trials: a flat LR is the cleaner probe
                )
            except Exception as exc:
                if is_oom_error(exc):
                    LOGGER.warning(
                        "[%s/%s] OOM at lr=%.1e; skipping.", arena.name, optimizer_name, lr
                    )
                    self._free()
                    continue
                raise

            value = res.primary_value
            trials.append(
                {"optimizer": optimizer_name, "lr": lr, "value": value, "failed": res.failed}
            )
            ok = math.isfinite(value) and not res.failed
            better = ok and (
                (arena.primary_mode == "max" and value > best_value)
                or (arena.primary_mode == "min" and value < best_value)
            )
            LOGGER.info(
                "[%s/%s] lr=%.1e -> %s=%.4f%s",
                arena.name,
                optimizer_name,
                lr,
                arena.primary_metric,
                value,
                "  <-- best" if better else "",
            )
            if better:
                best_value, best_lr = value, lr

        LOGGER.info(
            "[%s/%s] selected lr=%.2e (%s=%.4f)",
            arena.name,
            optimizer_name,
            best_lr,
            arena.primary_metric,
            best_value,
        )
        return best_lr, trials

    def _free(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Stage 2 + orchestration
# --------------------------------------------------------------------------- #


class BenchmarkRunner:
    """End-to-end driver: sweep, multi-seed training, statistics and export."""

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        setup_console_logging()
        self.device = resolve_device(cfg.hardware.device)
        self.device_info = describe_device(self.device)
        self.gpu_label = format_device_label(self.device_info)
        if cfg.hardware.cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        os.makedirs(cfg.logging.output_dir, exist_ok=True)
        config_path = os.path.join(cfg.logging.output_dir, "config.json")
        save_config_with_hardware(cfg, config_path, self.device_info)
        self.step_csv = CSVLogger(os.path.join(cfg.logging.output_dir, "steps.csv"))
        self.summary_csv = CSVLogger(os.path.join(cfg.logging.output_dir, "summary.csv"))
        self.aggregate_csv = CSVLogger(os.path.join(cfg.logging.output_dir, "aggregate.csv"))
        self.stats_csv = CSVLogger(os.path.join(cfg.logging.output_dir, "significance.csv"))
        self.sweep_csv = CSVLogger(os.path.join(cfg.logging.output_dir, "lr_sweep.csv"))

    # ----------------------------- public API -------------------------- #

    def run(self) -> None:
        """Run every requested arena end-to-end."""
        log_device_banner(self.device_info)
        if self.cfg.offline:
            root = (
                self.cfg.arena_config(self.cfg.arenas[0]).data_root if self.cfg.arenas else "./data"
            )
            LOGGER.info("Offline mode: datasets must exist under %s", os.path.abspath(root))
        LOGGER.info("optimizers=%s | arenas=%s", self.cfg.optimizers, self.cfg.arenas)
        for arena_name in self.cfg.arenas:
            try:
                self.run_arena(arena_name)
            except Exception as exc:
                LOGGER.exception("Arena '%s' failed: %s", arena_name, exc)
        self._finalize_plots()
        LOGGER.info("Benchmark complete. Results in %s", self.cfg.logging.output_dir)

    def run_arena(self, arena_name: str) -> None:
        """Run Stage 1 + Stage 2 for one arena and record all results."""
        acfg = self.cfg.arena_config(arena_name)
        arena = build_arena(
            arena_name,
            data_root=acfg.data_root,
            batch_size=acfg.batch_size,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=self.cfg.hardware.pin_memory,
            synthetic=self.cfg.synthetic,
            offline=self.cfg.offline,
            extra=acfg.extra,
        )
        LOGGER.info("=== Arena: %s ===", arena_name)
        arena.prepare()

        # Build datasets once; per-run loaders are seeded copies (reproducible).
        base_train, base_val = arena.build_dataloaders()
        train_dataset, val_dataset = base_train.dataset, base_val.dataset
        engine = TrainEngine(
            arena, self.cfg, self.device, train_dataset, val_dataset, self.device_info
        )

        # ---- Stage 1: LR sweep (or use fixed LRs) ----
        # The sweep seed must be disjoint from the Stage-2 evaluation seeds:
        # reusing an evaluation seed for LR selection would let that seed's
        # data order / init influence both which LR is picked *and* the
        # reported score, biasing the final comparison in the selected
        # optimizer's favor.
        sweep_seed = self._resolve_sweep_seed()
        set_seed(sweep_seed)
        sweep_init = self._snapshot_init(arena)
        best_lrs = self._resolve_lrs(arena, engine, sweep_init, sweep_seed)

        # ---- Stage 2: multi-seed evaluation with identical per-seed init ----
        # Keyed by seed (not appended positionally) so that a run missing a
        # metric key (e.g. a failed run whose fallback _evaluate() only
        # returns the primary metric) can't silently shift later seeds out
        # of alignment between optimizers -- the paired significance test
        # below depends on a[i]/b[i] being the *same seed* for every i.
        per_opt_finals: dict[str, dict[str, dict[int, float]]] = {
            opt: {} for opt in self.cfg.optimizers
        }
        for seed in self.cfg.seeds:
            set_seed(seed)
            init_state = self._snapshot_init(arena)  # shared by all optimizers
            for optimizer_name in self.cfg.optimizers:
                lr = best_lrs[optimizer_name]
                res = self._run_with_oom_retry(engine, arena, optimizer_name, lr, seed, init_state)
                self._record_run(res)
                for metric, value in res.final_metrics.items():
                    per_opt_finals[optimizer_name].setdefault(metric, {})[seed] = value

        self._record_statistics(arena_name, per_opt_finals, best_lrs)

    # ----------------------------- internals --------------------------- #

    def _resolve_sweep_seed(self) -> int:
        """Pick the Stage-1 LR-sweep seed, guaranteed disjoint from ``cfg.seeds``.

        Reusing one of the Stage-2 evaluation seeds for LR selection would let
        that seed's data order/init influence both which LR is chosen and the
        reported score, biasing the final comparison (see issue #23). An
        explicit ``cfg.sweep_seed`` is honored as long as it doesn't collide
        with an evaluation seed; otherwise we derive one deterministically.
        """
        explicit = getattr(self.cfg, "sweep_seed", None)
        if explicit is not None:
            if explicit in self.cfg.seeds:
                raise ValueError(
                    f"cfg.sweep_seed={explicit} collides with an evaluation "
                    f"seed in cfg.seeds={self.cfg.seeds}; they must be disjoint."
                )
            return explicit

        # Deterministic fallback: smallest non-negative integer not already
        # used as an evaluation seed, starting just past the largest one so
        # sweep results stay stable if cfg.seeds gains entries later.
        candidate = max(self.cfg.seeds, default=-1) + 1
        while candidate in self.cfg.seeds:
            candidate += 1
        return candidate

    def _snapshot_init(self, arena: Arena) -> dict[str, torch.Tensor]:
        """Build a model under the current RNG and snapshot its initial weights."""
        model = arena.build_model()
        return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    def _resolve_lrs(
        self, arena: Arena, engine: TrainEngine, init_state, sweep_seed: int
    ) -> dict[str, float]:
        fixed = self.cfg.fixed_lrs.get(arena.name, {})
        sweeper = LRSweeper(engine, self.cfg)
        best_lrs: dict[str, float] = {}
        for optimizer_name in self.cfg.optimizers:
            if optimizer_name in fixed:
                best_lrs[optimizer_name] = float(fixed[optimizer_name])
                LOGGER.info(
                    "[%s/%s] using fixed lr=%.2e (sweep skipped).",
                    arena.name,
                    optimizer_name,
                    best_lrs[optimizer_name],
                )
                continue
            best_lr, trials = sweeper.sweep(optimizer_name, init_state, sweep_seed)
            best_lrs[optimizer_name] = best_lr
            for t in trials:
                self.sweep_csv.append(
                    {
                        "arena": arena.name,
                        "gpu_name": self.device_info.get("gpu_name"),
                        "device": self.device_info.get("device"),
                        **t,
                        "selected": t["lr"] == best_lr,
                    }
                )
        return best_lrs

    def _run_with_oom_retry(
        self,
        engine: TrainEngine,
        arena: Arena,
        optimizer_name: str,
        lr: float,
        seed: int,
        init_state,
    ) -> RunResult:
        """Run a job, halving the batch size on CUDA OOM (up to 2 retries)."""
        attempts = 0
        original_bs = arena.batch_size
        while True:
            try:
                res = engine.run(
                    optimizer_name,
                    lr,
                    seed,
                    init_state,
                    max_steps=self.cfg.train.max_steps,
                    max_epochs=self.cfg.train.max_epochs,
                    eval_every=self.cfg.train.eval_every,
                    logger=self._make_logger(arena.name, optimizer_name, f"seed{seed}", lr),
                    record_history=True,
                    use_scheduler=True,
                )
                arena.batch_size = original_bs
                return res
            except Exception as exc:
                if not is_oom_error(exc) or attempts >= 2 or arena.batch_size <= 1:
                    arena.batch_size = original_bs
                    if is_oom_error(exc):
                        LOGGER.error(
                            "[%s/%s/seed%d] OOM not recoverable; recording failure.",
                            arena.name,
                            optimizer_name,
                            seed,
                        )
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return RunResult(
                            arena.name, optimizer_name, seed, lr, failed=True, error="cuda-oom"
                        )
                    raise
                attempts += 1
                arena.batch_size = max(1, arena.batch_size // 2)
                LOGGER.warning(
                    "[%s/%s/seed%d] CUDA OOM; retrying with batch_size=%d (attempt %d).",
                    arena.name,
                    optimizer_name,
                    seed,
                    arena.batch_size,
                    attempts,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _make_logger(self, arena: str, optimizer: str, tag: str, lr: float) -> ExperimentLogger:
        return ExperimentLogger(
            output_dir=self.cfg.logging.output_dir,
            arena=arena,
            optimizer=optimizer,
            tag=tag,
            use_tb=self.cfg.logging.tensorboard,
            use_wandb=self.cfg.logging.wandb,
            wandb_project=self.cfg.logging.wandb_project,
            wandb_entity=self.cfg.logging.wandb_entity,
            config={
                "arena": arena,
                "optimizer": optimizer,
                "lr": lr,
                "tag": tag,
                "gpu_name": self.device_info.get("gpu_name"),
                "gpu_vram_gb": self.device_info.get("gpu_vram_gb"),
                "device": self.device_info.get("device"),
            },
            device_info=self.device_info,
        )

    def _record_run(self, res: RunResult) -> None:
        hw = {
            "gpu_name": self.device_info.get("gpu_name"),
            "device": self.device_info.get("device"),
        }
        for row in res.step_history:
            self.step_csv.append(
                {
                    "arena": res.arena,
                    "optimizer": res.optimizer,
                    "seed": res.seed,
                    "lr": res.lr,
                    **hw,
                    **row,
                }
            )
        self.summary_csv.append(
            {
                "arena": res.arena,
                "optimizer": res.optimizer,
                "seed": res.seed,
                "lr": res.lr,
                "failed": res.failed,
                "error": res.error or "",
                **hw,
                **{f"final_{k}": v for k, v in res.final_metrics.items()},
            }
        )

    def _record_statistics(
        self,
        arena_name: str,
        per_opt_finals: dict[str, dict[str, dict[int, float]]],
        best_lrs: dict[str, float],
    ) -> None:
        """Compute Mean+/-Std per optimizer and paired, corrected significance tests."""
        # Aggregate (Mean +/- Std) over seeds.
        all_metrics = sorted({m for d in per_opt_finals.values() for m in d})
        for optimizer_name, metric_map in per_opt_finals.items():
            row: dict[str, Any] = {
                "arena": arena_name,
                "optimizer": optimizer_name,
                "lr": best_lrs.get(optimizer_name, float("nan")),
                "gpu_name": self.device_info.get("gpu_name"),
                "device": self.device_info.get("device"),
            }
            for metric in all_metrics:
                agg: Aggregate = aggregate(list(metric_map.get(metric, {}).values()))
                row[f"{metric}_mean"] = agg.mean
                row[f"{metric}_std"] = agg.std
                row[f"{metric}_n"] = agg.n
            self.aggregate_csv.append(row)

        # Significance: reference optimizer vs each baseline, per metric.
        # Runs share seeds/inits across optimizers (see run_arena), so each
        # comparison is paired on seed rather than treated as two
        # independent samples. All comparisons for this arena form one
        # family and get a single Holm-Bonferroni correction, since the
        # table is read row-by-row as individual pairwise claims.
        ref = REFERENCE_OPTIMIZER
        if ref not in per_opt_finals:
            return

        pending: list[dict[str, Any]] = []
        raw_pvalues: list[float] = []
        for optimizer_name, metric_map in per_opt_finals.items():
            if optimizer_name == ref:
                continue
            for metric in all_metrics:
                a_by_seed = per_opt_finals[ref].get(metric, {})
                b_by_seed = metric_map.get(metric, {})
                common_seeds = sorted(set(a_by_seed) & set(b_by_seed))
                a = [a_by_seed[s] for s in common_seeds]
                b = [b_by_seed[s] for s in common_seeds]
                tt = paired_ttest(a, b)
                pending.append(
                    {
                        "arena": arena_name,
                        "metric": metric,
                        "reference": ref,
                        "baseline": optimizer_name,
                        f"{ref}_mean": aggregate(a).mean,
                        f"{optimizer_name}_mean": aggregate(b).mean,
                        "n_pairs": tt.n_pairs,
                        "t_stat": tt.t_stat,
                        "p_value": tt.p_value,
                        "df": tt.df,
                        "cohens_dz": tt.cohens_dz,
                        "significant_raw_p<0.05": tt.significant,
                    }
                )
                raw_pvalues.append(tt.p_value)

        adjusted, reject = holm_bonferroni(raw_pvalues, alpha=0.05)
        for row, p_holm, sig_holm in zip(pending, adjusted, reject):
            row["p_value_holm"] = p_holm
            row["significant_holm_p<0.05"] = sig_holm
            self.stats_csv.append(row)

    def _finalize_plots(self) -> None:
        if not self.cfg.logging.plots:
            return
        from .plotting import plot_learning_curves

        plot_dir = os.path.join(self.cfg.logging.output_dir, "plots")
        for split in ("train", "val"):
            written = plot_learning_curves(self.step_csv.path, plot_dir, split=split)
            for path in written:
                LOGGER.info("Wrote plot %s", path)
