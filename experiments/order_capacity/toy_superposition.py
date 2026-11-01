"""Path dependence of superposition geometry (H1), toy scale.

Question
    After a feature group has been learned on its own, does it keep dedicated
    dimensions once a second, identical group is mixed in?

Model
    Elhage et al. tied ReLU autoencoder. Features are i.i.d., so the only
    asymmetry between groups is which one is introduced first.

        h = x W^T
        y = ReLU(h W + b)

    W is (n_hidden, n_features). Feature dimensionality follows Elhage:

        D_i = ||W_i||^2 / sum_j (W_hat_i · W_j)^2

    with D_i = 0 when the column is numerically unlearned. Dedicated basis
    vectors score 1. An antipodal pair scores 1/2 each.

Lag index
    Λ = (D_early - D_late) / (D_early + D_late)

    D_early is the mean dimensionality of the group trained in phase 1.
    The simultaneous schedule has no order; its group-A vs group-B Λ is the
    seed-noise baseline the ordered schedules have to clear.

Go / no-go (fixed before looking at ordered outcomes)
    Let σ be the across-seed standard deviation of the simultaneous baseline.
    An ordered schedule clears the bar when both of these hold:

        mean(Λ)     > max(0.15, mean(|Λ_sim|) + 2 σ_Λ)
        mean(ΔD)    > max(0.10, 2 σ_ΔD)

    Outcome "go" requires BOTH introduction orders (A then B, and B then A)
    to clear the bar at the end of mixed training. Phase 2 is eight times
    as long as phase 1, so an effect that is only the phase-1 head start in
    sample count has a long window to wash out.

    "invalid" means phase 1 never gave the early group dedicated dimensions,
    so the mixture phase is not a test of retention.

Primary cell
    32 features, 24 hidden dimensions, feature probability 0.05.
    24 dimensions can hold 16 dedicated directions and still leave 8
    dimensions in which the second group can be packed. That is the geometry
    H1 names: early features keep private dimensions, later features are
    squeezed into superposition. A 16-dimensional bottleneck cannot do this
    without the early features giving dimensions back; that cell is only a
    sensitivity run.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

NORM_FLOOR = 0.05
LAG_PRACTICAL = 0.15
GAP_PRACTICAL = 0.10
PHASE1_DEDICATED_MIN = 0.75


@dataclass(frozen=True)
class Config:
    n_features: int = 32
    n_hidden: int = 24
    feature_prob: float = 0.05
    batch_size: int = 2048
    lr: float = 1e-3
    phase1_steps: int = 1000
    phase2_steps: int = 8000
    log_every: int = 200
    init_std: float = 0.01
    eval_size: int = 16384
    eval_seed: int = 12345

    @property
    def total_steps(self) -> int:
        return self.phase1_steps + self.phase2_steps


@dataclass(frozen=True)
class Decision:
    outcome: str
    threshold_lag: float
    threshold_gap: float
    phase1_dedicated: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def feature_dimensionality(weights: torch.Tensor, norm_floor: float = NORM_FLOOR):
    """Elhage feature dimensionality. `weights` is (n_hidden, n_features)."""
    norms = weights.norm(dim=0)
    hat = weights / norms.clamp(min=1e-8)
    dots = hat.T @ weights
    dimensionality = (norms ** 2) / dots.pow(2).sum(dim=1).clamp(min=1e-12)
    dimensionality = torch.where(norms < norm_floor, torch.zeros_like(dimensionality), dimensionality)
    return dimensionality, norms


def lag_index(d_early: float, d_late: float, eps: float = 1e-8) -> float:
    return (d_early - d_late) / (d_early + d_late + eps)


def decide(
    effect_ab: float,
    effect_ba: float,
    gap_ab: float,
    gap_ba: float,
    sigma_lag: float,
    sigma_gap: float,
    mean_abs_lag_sim: float,
    phase1_dedicated: float,
) -> Decision:
    threshold_lag = max(LAG_PRACTICAL, mean_abs_lag_sim + 2.0 * sigma_lag)
    threshold_gap = max(GAP_PRACTICAL, 2.0 * sigma_gap)
    if phase1_dedicated < PHASE1_DEDICATED_MIN:
        return Decision(
            outcome="invalid",
            threshold_lag=threshold_lag,
            threshold_gap=threshold_gap,
            phase1_dedicated=phase1_dedicated,
            reason="phase 1 did not give the early group dedicated dimensions",
        )
    ab_ok = effect_ab > threshold_lag and gap_ab > threshold_gap
    ba_ok = effect_ba > threshold_lag and gap_ba > threshold_gap
    if ab_ok and ba_ok:
        reason = "both introduction orders still exceed the simultaneous seed-noise baseline after mixed training"
        outcome = "go"
    else:
        reason = "the order effect does not clear the seed-noise baseline in both introduction orders"
        outcome = "no_go"
    return Decision(
        outcome=outcome,
        threshold_lag=threshold_lag,
        threshold_gap=threshold_gap,
        phase1_dedicated=phase1_dedicated,
        reason=reason,
    )


def reconstruct(weights: torch.Tensor, bias: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    hidden = features @ weights.T
    return torch.relu(hidden @ weights + bias)


def sample_features(batch: int, n_features: int, feature_prob: float) -> torch.Tensor:
    active = torch.rand(batch, n_features) < feature_prob
    values = torch.rand(batch, n_features)
    return active.float() * values


def make_eval_set(config: Config) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config.eval_seed)
    active = torch.rand(config.eval_size, config.n_features, generator=generator) < config.feature_prob
    values = torch.rand(config.eval_size, config.n_features, generator=generator)
    return active.float() * values


def zero_predictor_loss(feature_prob: float) -> float:
    """E[(x - 0)^2] for x = Bernoulli(p) * Uniform(0, 1)."""
    return feature_prob / 3.0


def group_indices(n_features: int, schedule: str) -> tuple[torch.Tensor, torch.Tensor]:
    if n_features % 2 != 0:
        raise ValueError("n_features must be even so the two groups have equal size")
    group_a = torch.arange(0, n_features // 2)
    group_b = torch.arange(n_features // 2, n_features)
    if schedule == "ab":
        return group_a, group_b
    if schedule == "ba":
        return group_b, group_a
    if schedule == "sim":
        return group_a, group_b
    raise ValueError(f"unknown schedule {schedule}")


@torch.no_grad()
def evaluate(weights: torch.Tensor, bias: torch.Tensor, eval_x: torch.Tensor) -> dict:
    prediction = reconstruct(weights, bias, eval_x)
    per_feature_loss = (prediction - eval_x).pow(2).mean(dim=0)
    dimensionality, norms = feature_dimensionality(weights)
    return {
        "loss": per_feature_loss.detach().cpu(),
        "dimensionality": dimensionality.detach().cpu(),
        "norms": norms.detach().cpu(),
    }


def _mean(values: torch.Tensor, index: torch.Tensor) -> float:
    return float(values[index].mean())


def summarize_checkpoint(metrics: dict, early: torch.Tensor, late: torch.Tensor, step: int) -> dict:
    d_early = _mean(metrics["dimensionality"], early)
    d_late = _mean(metrics["dimensionality"], late)
    loss_early = _mean(metrics["loss"], early)
    loss_late = _mean(metrics["loss"], late)
    return {
        "step": step,
        "d_early": d_early,
        "d_late": d_late,
        "d_gap": d_early - d_late,
        "lag": lag_index(d_early, d_late),
        "loss_early": loss_early,
        "loss_late": loss_late,
        "loss_gap": loss_late - loss_early,
        "norm_early": _mean(metrics["norms"], early),
        "norm_late": _mean(metrics["norms"], late),
        "sum_dimensionality": float(metrics["dimensionality"].sum()),
    }


def train_one(config: Config, seed: int, schedule: str, eval_x: torch.Tensor) -> dict:
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(torch.randn(config.n_hidden, config.n_features) * config.init_std)
    bias = torch.nn.Parameter(torch.zeros(config.n_features))
    optimizer = torch.optim.Adam([weights, bias], lr=config.lr)
    early, late = group_indices(config.n_features, schedule)
    history = []

    def record(step: int) -> None:
        history.append(summarize_checkpoint(evaluate(weights, bias, eval_x), early, late, step))

    record(0)
    for step in range(config.total_steps):
        features = sample_features(config.batch_size, config.n_features, config.feature_prob)
        include = torch.ones(config.n_features, dtype=torch.bool)
        in_phase1 = schedule != "sim" and step < config.phase1_steps
        if in_phase1:
            features = features.clone()
            features[:, late] = 0
            include[late] = False
        optimizer.zero_grad(set_to_none=True)
        prediction = reconstruct(weights, bias, features)
        loss = (prediction - features).pow(2)[:, include].mean()
        loss.backward()
        optimizer.step()
        done = step + 1
        if done % config.log_every == 0 or done == config.phase1_steps or done == config.total_steps:
            record(done)
    return {
        "seed": seed,
        "schedule": schedule,
        "history": history,
        "final_dimensionality": feature_dimensionality(weights.detach())[0].cpu().tolist(),
    }


def _at_step(history: list[dict], step: int) -> dict:
    for row in history:
        if row["step"] == step:
            return row
    raise KeyError(step)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0] if values else float("nan"), float("nan"))
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def endpoint_table(runs: list[dict], step: int) -> dict:
    by_schedule: dict[str, list[dict]] = {}
    for run in runs:
        by_schedule.setdefault(run["schedule"], []).append(_at_step(run["history"], step))
    table = {}
    for schedule, rows in by_schedule.items():
        summary = {"n": len(rows)}
        for key in ("d_early", "d_late", "d_gap", "lag", "loss_early", "loss_late", "loss_gap", "norm_early", "norm_late", "sum_dimensionality"):
            mean, std = _mean_std([row[key] for row in rows])
            summary[key] = mean
            summary[f"{key}_std"] = std
        table[schedule] = summary
    return table


def geometry_label(d_early: float, d_late: float, loss_late: float, zero_loss: float) -> str:
    late_learned = loss_late < 0.5 * zero_loss and d_late > 0.15
    early_dedicated = d_early >= PHASE1_DEDICATED_MIN
    if early_dedicated and late_learned and d_late < PHASE1_DEDICATED_MIN:
        return "packing"
    if early_dedicated and not late_learned:
        return "exclusion"
    if abs(d_early - d_late) <= GAP_PRACTICAL:
        return "symmetric"
    return "other"


def analyze(runs: list[dict], config: Config) -> dict:
    end = config.total_steps
    phase1 = config.phase1_steps
    table = endpoint_table(runs, end)
    phase1_table = endpoint_table(runs, phase1)
    sim_gaps = [
        _at_step(run["history"], end)["d_gap"]
        for run in runs
        if run["schedule"] == "sim"
    ]
    sim_lags = [
        _at_step(run["history"], end)["lag"]
        for run in runs
        if run["schedule"] == "sim"
    ]
    _, sigma_gap = _mean_std(sim_gaps)
    _, sigma_lag = _mean_std(sim_lags)
    mean_abs_lag_sim = sum(abs(v) for v in sim_lags) / len(sim_lags)
    phase1_dedicated = 0.5 * (
        phase1_table["ab"]["d_early"] + phase1_table["ba"]["d_early"]
    )
    decision = decide(
        effect_ab=table["ab"]["lag"],
        effect_ba=table["ba"]["lag"],
        gap_ab=table["ab"]["d_gap"],
        gap_ba=table["ba"]["d_gap"],
        sigma_lag=sigma_lag,
        sigma_gap=sigma_gap,
        mean_abs_lag_sim=mean_abs_lag_sim,
        phase1_dedicated=phase1_dedicated,
    )
    zero_loss = zero_predictor_loss(config.feature_prob)
    geometry = {
        schedule: geometry_label(
            table[schedule]["d_early"],
            table[schedule]["d_late"],
            table[schedule]["loss_late"],
            zero_loss,
        )
        for schedule in ("ab", "ba", "sim")
    }
    # For the simultaneous run the "late" label is just group B, not a curriculum.
    return {
        "decision": decision.as_dict(),
        "geometry": geometry,
        "endpoint": table,
        "phase1": phase1_table,
        "noise_baseline": {
            "sigma_gap_sim": sigma_gap,
            "sigma_lag_sim": sigma_lag,
            "mean_abs_lag_sim": mean_abs_lag_sim,
            "mean_gap_sim": sum(sim_gaps) / len(sim_gaps),
        },
        "zero_predictor_loss": zero_loss,
        "snapshots": {
            str(step): endpoint_table(runs, step)
            for step in _snapshot_steps(config)
            if any(row["step"] == step for row in runs[0]["history"])
        },
    }


def _snapshot_steps(config: Config) -> list[int]:
    anchors = [
        config.phase1_steps,
        config.phase1_steps + config.phase1_steps,
        config.phase1_steps + 3 * config.phase1_steps,
        config.total_steps,
    ]
    return sorted({step for step in anchors if 0 < step <= config.total_steps})


def plot_curves(runs: list[dict], config: Config, path: Path) -> None:
    schedules = ("ab", "ba", "sim")
    colors = {"ab": "#0B6E4F", "ba": "#B45309", "sim": "#3F3F46"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)

    for schedule in schedules:
        selected = [run for run in runs if run["schedule"] == schedule]
        steps = [row["step"] for row in selected[0]["history"]]
        for axis, key, ylabel in (
            (axes[0], "d_gap", "Dimensionality gap  D_early − D_late"),
            (axes[1], "lag", "Lag index  Λ"),
        ):
            series = [[row[key] for row in run["history"]] for run in selected]
            mean = [sum(col) / len(col) for col in zip(*series)]
            std = []
            for col in zip(*series):
                mu = sum(col) / len(col)
                var = sum((v - mu) ** 2 for v in col) / max(len(col) - 1, 1)
                std.append(math.sqrt(var))
            xs = steps
            axis.plot(xs, mean, color=colors[schedule], label=schedule, linewidth=1.8)
            axis.fill_between(
                xs,
                [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color=colors[schedule],
                alpha=0.15,
                linewidth=0,
            )
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Training step")
            axis.axvline(config.phase1_steps, color="#A1A1AA", linestyle="--", linewidth=1)
            axis.axhline(0.0, color="#D4D4D8", linewidth=1)

    axes[0].set_title("Capacity gap")
    axes[1].set_title("Lag index")
    axes[1].legend(frameon=False, title="schedule")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_endpoint(analysis: dict, path: Path) -> None:
    schedules = ("ab", "ba", "sim")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    import numpy as np

    x = np.arange(len(schedules))
    width = 0.36
    early = [analysis["endpoint"][s]["d_early"] for s in schedules]
    late = [analysis["endpoint"][s]["d_late"] for s in schedules]
    early_err = [analysis["endpoint"][s]["d_early_std"] for s in schedules]
    late_err = [analysis["endpoint"][s]["d_late_std"] for s in schedules]
    ax.bar(x - width / 2, early, width, yerr=early_err, label="early / group A", color="#0F766E", capsize=3)
    ax.bar(x + width / 2, late, width, yerr=late_err, label="late / group B", color="#C2410C", capsize=3)
    ax.set_xticks(x, schedules)
    ax.set_ylabel("Mean feature dimensionality")
    ax.set_ylim(0, 1.05)
    ax.set_title("After mixed training")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_schedule_grid(config: Config, seeds: list[int], schedules: tuple[str, ...]) -> list[dict]:
    eval_x = make_eval_set(config)
    runs = []
    for seed in seeds:
        for schedule in schedules:
            runs.append(train_one(config, seed, schedule, eval_x))
            last = runs[-1]["history"][-1]
            print(
                f"seed {seed} {schedule:3s}  "
                f"D_early {last['d_early']:.3f}  D_late {last['d_late']:.3f}  "
                f"Λ {last['lag']:.3f}  loss {last['loss_early']:.4f}/{last['loss_late']:.4f}",
                flush=True,
            )
    return runs


def slim_runs(runs: list[dict]) -> list[dict]:
    """Drop per-feature vectors so the saved curves stay readable."""
    return [
        {"seed": run["seed"], "schedule": run["schedule"], "history": run["history"]}
        for run in runs
    ]


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "primary"), default="primary")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--phase1-steps", type=int, default=None)
    parser.add_argument("--phase2-steps", type=int, default=None)
    parser.add_argument("--n-hidden", type=int, default=None)
    parser.add_argument("--feature-prob", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    values = {}
    if args.phase1_steps is not None:
        values["phase1_steps"] = args.phase1_steps
    if args.phase2_steps is not None:
        values["phase2_steps"] = args.phase2_steps
    if args.n_hidden is not None:
        values["n_hidden"] = args.n_hidden
    if args.feature_prob is not None:
        values["feature_prob"] = args.feature_prob
    if args.lr is not None:
        values["lr"] = args.lr
    if args.mode == "pilot":
        values.setdefault("phase1_steps", 2000)
        values.setdefault("phase2_steps", 2000)
        values["log_every"] = 100
    return Config(**values)


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    torch.set_num_threads(4)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "pilot":
        runs = run_schedule_grid(config, seeds=[0], schedules=("ab", "sim"))
        write_results(out / "pilot.json", {"config": asdict(config), "runs": slim_runs(runs)})
        print("pilot wrote", out / "pilot.json")
        return
    seeds = list(range(args.seeds))
    runs = run_schedule_grid(config, seeds=seeds, schedules=("ab", "ba", "sim"))
    analysis = analyze(runs, config)
    payload = {"config": asdict(config), "analysis": analysis, "runs": slim_runs(runs)}
    write_results(out / "primary.json", payload)
    plot_curves(runs, config, out / "lag_curves.png")
    plot_endpoint(analysis, out / "endpoint_dimensionality.png")
    print(json.dumps({"decision": analysis["decision"], "geometry": analysis["geometry"], "endpoint": analysis["endpoint"]}, indent=2))


if __name__ == "__main__":
    main()
