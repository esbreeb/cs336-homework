"""H3 toy: does the teacher's own acquisition order save student tokens?

Comparison, same student width, same data, same step cap:

    progressive     match the teacher's intermediate checkpoints, in order,
                    on every feature (the ICLR 2025 baseline)
    developmental   match the final teacher, but only on the features it
                    acquired earliest, adding the next group each stage
    reverse         same as developmental, latest-acquired features first
    joint           match the final teacher on every feature from step 0

The student is narrower than the teacher, so it cannot store every feature
in its own dimension. Evaluation is always full-feature MSE against the
final teacher, including features the current stage is not yet training.

Token metric
    steps until eval MSE falls below a quarter of the step-0 MSE.
    A half-way cut is too loose: it can be crossed while most features are
    still being withheld. Tokens = steps × batch size. Runs that never cross
    are censored at the cap.

Go, fixed before comparing schedules
    developmental beats progressive when all three hold:

        mean(steps_progressive − steps_developmental) > 2 × its standard error
        mean(steps_developmental) < 0.9 × mean(steps_progressive)
        fewer than half the seeds are censored in either schedule

    Censored runs enter the mean as the cap, which shrinks a difference when
    the slower schedule is the one that failed. Joint is reported and is not
    part of the go rule. The claim under test is against progressive
    distillation, not against ordinary distillation.

Invalid
    acquisition times do not form two waves separated by at least 400 steps.
    A wave is a run of features whose acquisition steps differ by less than
    that gap. Equal-sized bins are not used: features the teacher learns in
    the same window stay in the same stage.
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

from toy_superposition import feature_dimensionality, reconstruct, sample_features, zero_predictor_loss

WAVE_GAP = 400


@dataclass(frozen=True)
class EfficiencyConfig:
    n_features: int = 32
    teacher_hidden: int = 20
    student_hidden: int = 8
    feature_prob: float = 0.05
    batch_size: int = 2048
    lr: float = 1e-3
    teacher_steps: int = 2000
    stage_steps: int = 2000
    importance_decay: float = 0.25
    log_every: int = 100
    init_std: float = 0.01
    eval_size: int = 8192
    eval_seed: int = 12345
    teacher_seed: int = 0
    tier_size: int = 8


def importance_vector(config: EfficiencyConfig) -> torch.Tensor:
    importance = torch.empty(config.n_features)
    n_tiers = config.n_features // config.tier_size
    for tier in range(n_tiers):
        start = tier * config.tier_size
        importance[start : start + config.tier_size] = config.importance_decay ** tier
    return importance


def acquisition_order(loss_curve: list[tuple[int, list[float]]], acquired_below: float) -> list[int]:
    """Feature indices sorted by first step under `acquired_below`, then final loss, then index."""
    n_features = len(loss_curve[0][1])
    final = loss_curve[-1][1]
    ranked = []
    for index in range(n_features):
        acquired = None
        for step, losses in loss_curve:
            if losses[index] < acquired_below:
                acquired = step
                break
        ranked.append((acquired if acquired is not None else 10**9, final[index], index))
    ranked.sort()
    return [index for _, _, index in ranked]


def cluster_waves(acquired_steps: list[int], min_gap: int = WAVE_GAP) -> list[list[int]]:
    """Group features into waves. A new wave starts when acquisition jumps by `min_gap`."""
    order = sorted(range(len(acquired_steps)), key=lambda index: (acquired_steps[index], index))
    waves = [[order[0]]]
    for previous, current in zip(order, order[1:]):
        if acquired_steps[current] - acquired_steps[previous] >= min_gap:
            waves.append([])
        waves[-1].append(current)
    return waves


def wave_separation(waves: list[list[int]], acquired_steps: list[int]) -> float:
    if len(waves) < 2:
        return 0.0
    first = [acquired_steps[index] for index in waves[0]]
    last = [acquired_steps[index] for index in waves[-1]]
    return sum(last) / len(last) - sum(first) / len(first)


def interquartile_range(values: list[int]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    def q(p: float) -> float:
        pos = p * (n - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return float(ordered[lo])
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
    return q(0.75) - q(0.25)


@dataclass(frozen=True)
class EfficiencyDecision:
    outcome: str
    mean_steps_dev: float
    mean_steps_prog: float
    paired_difference: float
    paired_se: float
    censored_dev: int
    censored_prog: int
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def decide_efficiency(
    steps_dev: list[int],
    steps_prog: list[int],
    cap: int,
    censored_dev: int | None = None,
    censored_prog: int | None = None,
) -> EfficiencyDecision:
    if len(steps_dev) != len(steps_prog) or not steps_dev:
        raise ValueError("schedules must be paired seed by seed")
    diff = [p - d for d, p in zip(steps_dev, steps_prog)]
    mean_diff = sum(diff) / len(diff)
    var = sum((v - mean_diff) ** 2 for v in diff) / (len(diff) - 1)
    se = math.sqrt(var / len(diff))
    mean_dev = sum(steps_dev) / len(steps_dev)
    mean_prog = sum(steps_prog) / len(steps_prog)
    if censored_dev is None:
        censored_dev = sum(step >= cap for step in steps_dev)
    if censored_prog is None:
        censored_prog = sum(step >= cap for step in steps_prog)
    base = dict(
        mean_steps_dev=mean_dev,
        mean_steps_prog=mean_prog,
        paired_difference=mean_diff,
        paired_se=se,
        censored_dev=censored_dev,
        censored_prog=censored_prog,
    )
    if censored_dev >= len(steps_dev) / 2 or censored_prog >= len(steps_prog) / 2:
        return EfficiencyDecision(
            outcome="inconclusive",
            reason="half or more of the seeds never reached the loss target",
            **base,
        )
    faster = mean_diff > 2.0 * se and mean_dev < 0.9 * mean_prog
    if faster:
        reason = "developmental order reaches the target in fewer student steps than progressive distillation"
        outcome = "go"
    else:
        reason = "developmental order does not beat progressive distillation by the pre-registered margin"
        outcome = "no_go"
    return EfficiencyDecision(outcome=outcome, reason=reason, **base)


def _init_model(n_hidden: int, n_features: int, init_std: float):
    weights = torch.nn.Parameter(torch.randn(n_hidden, n_features) * init_std)
    bias = torch.nn.Parameter(torch.zeros(n_features))
    return weights, bias


@torch.no_grad()
def per_feature_data_loss(weights: torch.Tensor, bias: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    prediction = reconstruct(weights, bias, features)
    return (prediction - features).pow(2).mean(dim=0)


def train_teacher(config: EfficiencyConfig, eval_x: torch.Tensor) -> dict:
    torch.manual_seed(config.teacher_seed)
    weights, bias = _init_model(config.teacher_hidden, config.n_features, config.init_std)
    optimizer = torch.optim.Adam([weights, bias], lr=config.lr)
    importance = importance_vector(config)
    curve = []
    weight_snapshots = []

    def snapshot(step: int) -> None:
        losses = per_feature_data_loss(weights, bias, eval_x).tolist()
        curve.append((step, losses))
        weight_snapshots.append(
            {"step": step, "weights": weights.detach().clone(), "bias": bias.detach().clone()}
        )

    snapshot(0)
    for step in range(config.teacher_steps):
        features = sample_features(config.batch_size, config.n_features, config.feature_prob)
        optimizer.zero_grad(set_to_none=True)
        prediction = reconstruct(weights, bias, features)
        loss = ((prediction - features).pow(2) * importance).mean()
        loss.backward()
        optimizer.step()
        done = step + 1
        if done % config.log_every == 0 or done == config.teacher_steps:
            snapshot(done)
    acquired_below = 0.25 * zero_predictor_loss(config.feature_prob)
    order = acquisition_order(curve, acquired_below)
    acquired_steps = []
    for index in range(config.n_features):
        found = 10**9
        for step, losses in curve:
            if losses[index] < acquired_below:
                found = step
                break
        acquired_steps.append(found)
    groups = cluster_waves(acquired_steps, WAVE_GAP)
    n_waves = len(groups)
    checkpoint_at = [config.teacher_steps * (wave + 1) // n_waves for wave in range(n_waves)]
    checkpoints = []
    for wanted in checkpoint_at:
        checkpoints.append(min(weight_snapshots, key=lambda item: abs(item["step"] - wanted)))
    return {
        "weights": weights.detach(),
        "bias": bias.detach(),
        "checkpoints": checkpoints,
        "curve": curve,
        "order": order,
        "groups": groups,
        "acquired_steps": acquired_steps,
        "wave_separation": wave_separation(groups, acquired_steps),
        "n_unacquired": sum(step >= 10**9 for step in acquired_steps),
        "importance": importance.tolist(),
        "student_steps": config.stage_steps * n_waves,
    }


def _stage_include(schedule: str, stage: int, groups: list[list[int]], n_features: int) -> torch.Tensor:
    include = torch.zeros(n_features, dtype=torch.bool)
    if schedule in ("progressive", "joint"):
        include[:] = True
        return include
    chosen = groups[: stage + 1] if schedule == "developmental" else list(reversed(groups))[: stage + 1]
    for group in chosen:
        include[group] = True
    return include


def _teacher_for_stage(schedule: str, stage: int, teacher: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if schedule == "progressive":
        checkpoint = teacher["checkpoints"][stage]
        return checkpoint["weights"], checkpoint["bias"]
    return teacher["weights"], teacher["bias"]


@torch.no_grad()
def student_eval(weights, bias, teacher_w, teacher_b, eval_x: torch.Tensor) -> float:
    target = reconstruct(teacher_w, teacher_b, eval_x)
    prediction = reconstruct(weights, bias, eval_x)
    return float((prediction - target).pow(2).mean())


def train_student(config: EfficiencyConfig, seed: int, schedule: str, teacher: dict, eval_x: torch.Tensor) -> dict:
    torch.manual_seed(1000 + seed)
    weights, bias = _init_model(config.student_hidden, config.n_features, config.init_std)
    optimizer = torch.optim.Adam([weights, bias], lr=config.lr)
    final_w, final_b = teacher["weights"], teacher["bias"]
    n_stages = len(teacher["groups"])
    total_steps = config.stage_steps * n_stages
    history = [{"step": 0, "loss": student_eval(weights, bias, final_w, final_b, eval_x)}]
    target_loss = 0.25 * history[0]["loss"]
    crossed = None
    for step in range(total_steps):
        stage = min(step // config.stage_steps, n_stages - 1)
        features = sample_features(config.batch_size, config.n_features, config.feature_prob)
        teacher_w, teacher_b = _teacher_for_stage(schedule, stage, teacher)
        include = _stage_include(schedule, stage, teacher["groups"], config.n_features)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            target = reconstruct(teacher_w, teacher_b, features)
        prediction = reconstruct(weights, bias, features)
        loss = (prediction - target).pow(2)[:, include].mean()
        loss.backward()
        optimizer.step()
        done = step + 1
        if done % config.log_every == 0 or done == total_steps:
            eval_loss = student_eval(weights, bias, final_w, final_b, eval_x)
            history.append({"step": done, "loss": eval_loss})
            if crossed is None and eval_loss < target_loss:
                crossed = done
    if crossed is None:
        crossed = total_steps
    final_dim, _ = feature_dimensionality(weights.detach())
    return {
        "seed": seed,
        "schedule": schedule,
        "history": history,
        "steps_to_target": crossed,
        "censored": crossed >= total_steps and history[-1]["loss"] >= target_loss,
        "target_loss": target_loss,
        "final_loss": history[-1]["loss"],
        "final_sum_dimensionality": float(final_dim.sum()),
    }


def _mean_se(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def analyze_students(runs: list[dict], config: EfficiencyConfig, teacher: dict) -> dict:
    by_schedule: dict[str, list[dict]] = {}
    for run in runs:
        by_schedule.setdefault(run["schedule"], []).append(run)
    for rows in by_schedule.values():
        rows.sort(key=lambda run: run["seed"])
    summary = {}
    for schedule, rows in by_schedule.items():
        steps = [run["steps_to_target"] for run in rows]
        finals = [run["final_loss"] for run in rows]
        mean_steps, se_steps = _mean_se(steps)
        mean_final, se_final = _mean_se(finals)
        summary[schedule] = {
            "mean_steps": mean_steps,
            "se_steps": se_steps,
            "mean_tokens": mean_steps * config.batch_size,
            "mean_final_loss": mean_final,
            "se_final_loss": se_final,
            "n_censored": sum(run["censored"] for run in rows),
            "steps": steps,
        }
    decision = decide_efficiency(
        summary["developmental"]["steps"],
        summary["progressive"]["steps"],
        teacher["student_steps"],
        censored_dev=summary["developmental"]["n_censored"],
        censored_prog=summary["progressive"]["n_censored"],
    )
    if teacher["wave_separation"] < WAVE_GAP or teacher["n_unacquired"] > config.n_features / 4:
        decision = EfficiencyDecision(
            outcome="invalid",
            mean_steps_dev=decision.mean_steps_dev,
            mean_steps_prog=decision.mean_steps_prog,
            paired_difference=decision.paired_difference,
            paired_se=decision.paired_se,
            censored_dev=decision.censored_dev,
            censored_prog=decision.censored_prog,
            reason="teacher acquisition times do not form two waves separated by 400 steps",
        )
    return {"decision": decision.as_dict(), "schedules": summary}


def plot_student_curves(runs: list[dict], path: Path) -> None:
    colors = {
        "developmental": "#0F766E",
        "progressive": "#1D4ED8",
        "reverse": "#C2410C",
        "joint": "#3F3F46",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for schedule, color in colors.items():
        selected = [run for run in runs if run["schedule"] == schedule]
        steps = [row["step"] for row in selected[0]["history"]]
        series = [[row["loss"] for row in run["history"]] for run in selected]
        mean = [sum(col) / len(col) for col in zip(*series)]
        ax.plot(steps, mean, color=color, label=schedule, linewidth=1.8)
    ax.set_xlabel("Student training step")
    ax.set_ylabel("MSE against final teacher")
    ax.set_title("Student match to the final teacher")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(config: EfficiencyConfig, seeds: list[int], schedules: tuple[str, ...]) -> tuple[dict, list[dict]]:
    generator = torch.Generator().manual_seed(config.eval_seed)
    active = torch.rand(config.eval_size, config.n_features, generator=generator) < config.feature_prob
    values = torch.rand(config.eval_size, config.n_features, generator=generator)
    eval_x = active.float() * values
    teacher = train_teacher(config, eval_x)
    print(
        f"teacher waves {len(teacher['groups'])}  separation {teacher['wave_separation']:.0f}  "
        f"unacquired {teacher['n_unacquired']}  checkpoints {[c['step'] for c in teacher['checkpoints']]}",
        flush=True,
    )
    print("groups", teacher["groups"], flush=True)
    runs = []
    for seed in seeds:
        for schedule in schedules:
            run_result = train_student(config, seed, schedule, teacher, eval_x)
            runs.append(run_result)
            print(
                f"seed {seed} {schedule:14s}  steps {run_result['steps_to_target']:5d}  "
                f"final {run_result['final_loss']:.5f}  censored {run_result['censored']}",
                flush=True,
            )
    return teacher, runs


def slim_teacher(teacher: dict) -> dict:
    curve = [{"step": step, "mean_loss": sum(losses) / len(losses)} for step, losses in teacher["curve"]]
    return {
        "order": teacher["order"],
        "groups": teacher["groups"],
        "acquired_steps": teacher["acquired_steps"],
        "wave_separation": teacher["wave_separation"],
        "student_steps": teacher["student_steps"],
        "n_unacquired": teacher["n_unacquired"],
        "importance": teacher["importance"],
        "mean_loss_curve": curve,
        "checkpoint_steps": [ckpt["step"] for ckpt in teacher["checkpoints"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = EfficiencyConfig()
    schedules = ("developmental", "progressive", "reverse", "joint")
    teacher, runs = run(config, list(range(args.seeds)), schedules)
    analysis = analyze_students(runs, config, teacher)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "teacher": slim_teacher(teacher),
        "analysis": analysis,
        "runs": [
            {
                "seed": run["seed"],
                "schedule": run["schedule"],
                "steps_to_target": run["steps_to_target"],
                "censored": run["censored"],
                "final_loss": run["final_loss"],
                "history": run["history"],
            }
            for run in runs
        ],
    }
    (args.out / "efficiency.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_student_curves(runs, args.out / "efficiency_curves.png")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
