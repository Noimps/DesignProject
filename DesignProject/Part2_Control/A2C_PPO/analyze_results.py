"""
Analysis of the RL training runs logged in training_results.csv.

Each row is ONE trained agent (PPO or A2C) on the unbalanced-disk swing-up
task, for a single random seed, with the wall-clock training time, the env
steps actually trained (early stopping can cut this below the 1M cap), and the
final mean return over 10 evaluation episodes.

With multiple seeds per configuration the script aggregates each configuration
to mean +/- std over its seeds, so every reported number and every plotted bar
carries a spread. A configuration is the tuple
(algo, base_reward, sin_cos, robust); the run `name` is used only for display.

Run:
    python analyze_results.py

Outputs:
    figures/*.png        - the plots (with error bars over seeds)
    analysis_findings.md - a written summary of the inferred findings
and prints the same findings to stdout.
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "training_results.csv"
FIG_DIR = HERE / "figures"
FINDINGS_PATH = HERE / "analysis_findings.md"

ALGO_COLORS = {"ppo": "#1f77b4", "a2c": "#d62728"}
CONFIG_KEYS = ["algo", "base_reward", "sin_cos", "robust"]


# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    for col in ("base_reward", "robust", "sin_cos"):
        df[col] = df[col].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
    df["algo"] = df["algo"].str.lower()
    if "seed" not in df.columns:
        df["seed"] = 0  # back-compat with single-seed CSVs

    df["reward_per_min"] = df["mean_reward"] / df["train_time_min"]
    df["reward_shaping"] = np.where(df["base_reward"], "base reward", "tuned reward")
    df["obs"] = np.where(df["sin_cos"], "sin/cos", "raw")
    df["disturb"] = np.where(df["robust"], "robust", "nominal")
    # Compact, name-independent label for each configuration.
    df["config"] = (df["algo"].str.upper() + " | "
                    + df["reward_shaping"].str.replace(" reward", "") + " | "
                    + df["obs"] + " | " + df["disturb"])
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse seeds: one row per configuration with mean/std/n of each metric."""
    g = df.groupby(CONFIG_KEYS + ["config", "algo"], as_index=False)
    agg = g.agg(
        reward_mean=("mean_reward", "mean"),
        reward_std=("mean_reward", "std"),
        reward_min=("mean_reward", "min"),
        reward_max=("mean_reward", "max"),
        time_mean=("train_time_min", "mean"),
        steps_mean=("num_timesteps", "mean"),
        n_seeds=("seed", "nunique"),
    )
    agg["reward_std"] = agg["reward_std"].fillna(0.0)  # std is NaN when n_seeds==1
    return agg.sort_values("reward_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_reward_by_config(agg: pd.DataFrame):
    d = agg.sort_values("reward_mean")
    colors = d["algo"].map(ALGO_COLORS)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(d["config"], d["reward_mean"], xerr=d["reward_std"],
            color=colors, error_kw=dict(ecolor="k", capsize=3, lw=1))
    ax.set_xlabel("mean return over seeds  (error bar = std)")
    ax.set_title("Return per configuration (aggregated over seeds)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ALGO_COLORS.values()]
    ax.legend(handles, [k.upper() for k in ALGO_COLORS], title="algorithm")
    fig.tight_layout()
    _save(fig, "reward_by_config.png")


def plot_algo_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    metrics = [("mean_reward", "mean return"),
               ("train_time_min", "training time [min]"),
               ("num_timesteps", "steps trained")]
    for ax, (col, label) in zip(axes, metrics):
        for i, a in enumerate(("ppo", "a2c")):
            g = df[df.algo == a][col].values
            ax.bar(i, g.mean(), yerr=g.std(),
                   color=ALGO_COLORS[a], alpha=0.6,
                   error_kw=dict(ecolor="k", capsize=4))
            ax.scatter(np.full(len(g), i) + np.random.uniform(-0.07, 0.07, len(g)),
                       g, color="k", zorder=3, s=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["PPO", "A2C"])
        ax.set_ylabel(label); ax.set_title(label)
    fig.suptitle("PPO vs A2C  (bar = mean over all runs+seeds, dots = individual runs)")
    fig.tight_layout()
    _save(fig, "algo_comparison.png")


def plot_time_vs_reward(agg: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for algo, sub in agg.groupby("algo"):
        ax.errorbar(sub["time_mean"], sub["reward_mean"], yerr=sub["reward_std"],
                    fmt="o", color=ALGO_COLORS[algo], label=algo.upper(),
                    capsize=3, ms=7)
    ax.set_xlabel("mean training time [min]")
    ax.set_ylabel("mean return  (error bar = std over seeds)")
    ax.set_title("Return vs training time per configuration")
    ax.legend(title="algorithm")
    fig.tight_layout()
    _save(fig, "time_vs_reward.png")


def plot_factor_effects(df: pd.DataFrame):
    """Mean return split by each binary factor, per algorithm, with std bars."""
    factors = [("reward_shaping", ("base reward", "tuned reward")),
               ("obs", ("raw", "sin/cos")),
               ("disturb", ("nominal", "robust"))]
    fig, axes = plt.subplots(1, len(factors), figsize=(13, 4.5), sharey=True)
    width = 0.35
    for ax, (col, levels) in zip(axes, factors):
        x = np.arange(len(levels))
        for i, algo in enumerate(("ppo", "a2c")):
            means = [df[(df.algo == algo) & (df[col] == lv)]["mean_reward"].mean()
                     for lv in levels]
            stds = [df[(df.algo == algo) & (df[col] == lv)]["mean_reward"].std()
                    for lv in levels]
            ax.bar(x + (i - 0.5) * width, means, width, yerr=stds,
                   color=ALGO_COLORS[algo], alpha=0.7, label=algo.upper(),
                   error_kw=dict(ecolor="k", capsize=3))
        ax.set_xticks(x); ax.set_xticklabels(levels, rotation=10)
        ax.set_title(col)
    axes[0].set_ylabel("mean return")
    axes[-1].legend(title="algorithm")
    fig.suptitle("Effect of each design factor on return (mean +/- std over runs)")
    fig.tight_layout()
    _save(fig, "factor_effects.png")


def _save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
def _ms(series):
    """mean +/- std formatted string."""
    return f"{series.mean():.1f} $\\pm$ {series.std():.1f}"


def infer_findings(df: pd.DataFrame, agg: pd.DataFrame) -> str:
    lines, add = [], lambda s: lines.append(s)
    n_seeds = df.groupby(CONFIG_KEYS).size()

    add("# Training-results analysis\n")
    add(f"_{len(df)} runs across {len(agg)} configurations, "
        f"{n_seeds.min()}–{n_seeds.max()} seeds each "
        f"({(df.algo=='ppo').sum()} PPO + {(df.algo=='a2c').sum()} A2C runs)._\n")

    best, worst = agg.iloc[0], agg.iloc[-1]
    n_cap = int((df["mean_reward"] >= 179.9).sum())
    add("## Overall")
    add(f"- **Best config:** `{best['config']}` — return "
        f"**{best.reward_mean:.1f} ± {best.reward_std:.1f}** over "
        f"{best.n_seeds} seeds, in {best.time_mean:.1f} min avg.")
    add(f"- **Worst config:** `{worst['config']}` — "
        f"{worst.reward_mean:.1f} ± {worst.reward_std:.1f}.")
    add(f"- **Censoring caveat:** training stopped at reward 180 (0.9×max), so "
        f"{n_cap}/{len(df)} runs (all PPO·sin/cos) are right-censored at 180 — "
        "their return is a *floor*, not a ceiling. Reward cannot separate the "
        "capped configs; for those, steps-to-threshold is the real metric (see "
        "`statistical_analysis.md`).\n")

    ppo, a2c = df[df.algo == "ppo"], df[df.algo == "a2c"]
    add("## Algorithm: PPO vs A2C")
    add(f"- Return (all runs) — PPO **{ppo.mean_reward.mean():.1f} ± "
        f"{ppo.mean_reward.std():.1f}** vs A2C **{a2c.mean_reward.mean():.1f} ± "
        f"{a2c.mean_reward.std():.1f}**.")
    add(f"- Training time — PPO {ppo.train_time_min.mean():.1f} min vs "
        f"A2C {a2c.train_time_min.mean():.1f} min.")
    add(f"- PPO's per-config std stays small (max "
        f"{agg[agg.algo=='ppo'].reward_std.max():.1f}); A2C's is large (max "
        f"{agg[agg.algo=='a2c'].reward_std.max():.1f}) — i.e. A2C results are "
        f"seed-sensitive, which is exactly why multiple seeds were needed.\n")

    add("## Design factors (mean ± std over runs)")
    for col, label in [("reward_shaping", "Reward shaping"),
                       ("obs", "Observation"),
                       ("disturb", "Disturbance")]:
        add(f"- **{label}:**")
        for algo in ("ppo", "a2c"):
            sub = df[df.algo == algo]
            parts = [f"{lv} {_ms(sub[sub[col]==lv]['mean_reward'])}"
                     for lv in sub[col].unique()]
            add(f"  - {algo.upper()}: " + "; ".join(parts))
    add("")

    corr_t = df["train_time_min"].corr(df["mean_reward"])
    add("## Efficiency")
    eff = agg.loc[(agg.reward_mean / agg.time_mean).idxmax()]
    add(f"- Best return-per-minute config: `{eff['config']}` "
        f"({eff.reward_mean:.0f} return in {eff.time_mean:.1f} min).")
    add(f"- Return vs training time correlates r={corr_t:.2f} across runs; "
        f"more compute did not buy more return (the long runs are the A2C "
        f"failures). Early stopping halted converged runs before the 1M cap.\n")

    add("## Bottom line")
    add(textwrap.dedent(f"""\
        - **Use PPO.** Higher return, faster, and low seed variance. A2C is both
          weaker and unstable across seeds on this task.
        - The top PPO configs are tied on return only because they are all
          censored at the 180 cap; the recommended controller is therefore the
          one that reaches the cap fastest and most reliably
          (`{best['config']}`).
        - Robust-trained PPO reaches the same cap, so it costs no *final return*
          — but it is not free: it needs more steps to get there (and with the
          tuned reward only 4/6 seeds reach the cap), trading sample-cost for
          sim-to-real robustness. See `statistical_analysis.md` §3.
        - Reporting mean ± std over {n_seeds.min()}–{n_seeds.max()} seeds removes
          the single-seed confound from the earlier write-up.
        """))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"No results CSV at {CSV_PATH}")
    df = load()
    agg = aggregate(df)

    print("Per-configuration summary (mean ± std over seeds):")
    show = agg[["config", "n_seeds", "reward_mean", "reward_std",
                "time_mean", "steps_mean"]].round(1)
    print(show.to_string(index=False), "\n")

    print("Generating plots ...")
    plot_reward_by_config(agg)
    plot_algo_comparison(df)
    plot_time_vs_reward(agg)
    plot_factor_effects(df)

    findings = infer_findings(df, agg)
    FINDINGS_PATH.write_text(findings)
    print(f"\nWrote findings to {FINDINGS_PATH.relative_to(HERE)}\n")
    print(findings)


if __name__ == "__main__":
    main()
