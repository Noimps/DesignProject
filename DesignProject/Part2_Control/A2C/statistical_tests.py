"""
Inferential statistics on the trained swing-up agents (training_results.csv).

The descriptive layer (analyze_results.py) reports mean +/- std bars. This
script adds the hypothesis tests that those bars only hint at, exploiting the
fact that the experiment is a *paired / blocked* design: the same six seeds
(42-47) were trained under matched conditions, so every comparison can be run
within seed-blocks instead of as independent samples. That removes seed as a
nuisance variable and is far more powerful at n=6/cell.

For each comparison we report:
  - the paired Wilcoxon signed-rank test  (primary: non-parametric, robust to
    the heavy A2C-failure skew),
  - the paired t-test                      (secondary, for reference),
  - Cohen's d_z                            (paired effect size),
  - the matched-pairs rank-biserial r      (Wilcoxon effect size),
  - a 95% bootstrap CI on the mean paired difference.

Because n is small (6 seeds), the smallest attainable two-sided Wilcoxon
p-value at 6 pairs is 0.031; comparisons pooled over several blocks (e.g. PPO
vs A2C over all 36 matched config,seed pairs) have far more resolution.

Run:
    python statistical_tests.py
Outputs:
    statistical_analysis.md      - the written report (also printed)
    figures/effect_sizes.png     - forest plot of mean diffs with 95% CIs
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "training_results.csv"
FIG_DIR = HERE / "figures"
OUT_PATH = HERE / "statistical_analysis.md"

KEYS = ["algo", "base_reward", "sin_cos", "robust", "seed"]
RNG = np.random.default_rng(0)
ALPHA = 0.05
N_BOOT = 10000
# Training used StopTrainingOnRewardThreshold(0.9 * max_eval_reward) with
# max_eval_reward = max_steps = 200, i.e. a hard stop at 180. Final mean_reward
# is therefore RIGHT-CENSORED at this value: a run that reaches it stops, so its
# reward is 180 by construction, not because that is its true ceiling. The
# discriminating metric for censored runs is steps-to-threshold, not reward.
REWARD_CAP = 180.0
CAP_TOL = 0.1  # treat reward >= CAP - TOL as "hit the cap"


# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    for col in ("base_reward", "robust", "sin_cos"):
        df[col] = df[col].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False})
    df["algo"] = df["algo"].str.lower()
    df["censored"] = df["mean_reward"] >= (REWARD_CAP - CAP_TOL)
    return df


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(diff: np.ndarray, n_boot=N_BOOT, alpha=ALPHA):
    """Percentile bootstrap CI for the mean of the paired differences."""
    idx = RNG.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cohen_dz(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for the Wilcoxon test."""
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rpos = r[d > 0].sum()
    rneg = r[d < 0].sum()
    total = rpos + rneg
    return float((rpos - rneg) / total) if total > 0 else 0.0


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare paired arrays a vs b (difference = a - b)."""
    diff = a - b
    out = dict(n=int(diff.size), mean_diff=float(diff.mean()),
               dz=cohen_dz(diff), rrb=rank_biserial(diff))
    out["ci_lo"], out["ci_hi"] = bootstrap_ci(diff)
    # paired t-test
    t = stats.ttest_rel(a, b)
    out["t_p"] = float(t.pvalue)
    # Wilcoxon (guard against all-zero / tiny n)
    if np.any(diff != 0) and diff.size >= 1:
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", method="auto")
            out["w_p"] = float(w.pvalue)
        except ValueError:
            out["w_p"] = np.nan
    else:
        out["w_p"] = 1.0
    return out


def paired_frame(df: pd.DataFrame, group_col, level_a, level_b, block_cols):
    """Pivot two levels of `group_col` onto matched rows defined by block_cols.
    Returns (a_values, b_values) aligned over the common blocks."""
    a = df[df[group_col] == level_a].set_index(block_cols)["mean_reward"]
    b = df[df[group_col] == level_b].set_index(block_cols)["mean_reward"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].sort_index(), b.loc[common].sort_index()
    return a.values, b.values


def stars(p):
    return ("***" if p < 1e-3 else "**" if p < 1e-2
            else "*" if p < 0.05 else "ns")


def fmt_d(dz):
    mag = abs(dz)
    tag = ("large" if mag >= 0.8 else "medium" if mag >= 0.5
           else "small" if mag >= 0.2 else "negligible")
    return f"{dz:+.2f} ({tag})"


# ---------------------------------------------------------------------------
def main():
    df = load()
    lines, add = [], lambda s: print(s) or lines.append(s)
    rows_for_plot = []  # (label, mean_diff, ci_lo, ci_hi)
    test_rows = []      # dicts for the LaTeX tests table

    add("# Statistical analysis of the trained swing-up agents\n")
    add(f"_{len(df)} runs, 12 configurations x 6 seeds (42-47). "
        "Paired/blocked design: comparisons are made within matched seed "
        "blocks. Primary test = paired Wilcoxon signed-rank; paired t-test "
        "and Cohen's d_z reported alongside; 95% CIs are percentile bootstrap "
        f"(B={N_BOOT}) on the mean paired difference._\n")

    # --- 0. Censoring warning --------------------------------------------
    n_cap = int(df.censored.sum())
    cap_by_algo = df.groupby("algo")["censored"].sum()
    add("> **Censoring caveat.** Training stopped on reaching reward "
        f"{REWARD_CAP:.0f} (0.9 x max). {n_cap}/{len(df)} runs hit that cap "
        f"(PPO {int(cap_by_algo.get('ppo',0))}, A2C {int(cap_by_algo.get('a2c',0))}) "
        "- all of them PPO+sin/cos configs. Their reward is right-censored at "
        "180, so reward-based comparisons *between censored configs* are "
        "uninformative (they are equal by construction); for those, the "
        "discriminating metric is **steps-to-threshold** (Section 3). "
        "Comparisons where one side is censored and the other is not "
        "(PPO vs A2C, sin/cos vs raw) are therefore **conservative**: the true "
        "gap is at least the reported one.\n")

    def report(title, a, b, name_a, name_b):
        r = paired_test(a, b)
        rows_for_plot.append((title, r["mean_diff"], r["ci_lo"], r["ci_hi"]))
        test_rows.append(dict(title=title, **r))
        add(f"### {title}")
        add(f"- pairs n = {r['n']}  ({name_a} - {name_b})")
        add(f"- mean difference = **{r['mean_diff']:+.1f}** return  "
            f"(95% CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}])")
        add(f"- Wilcoxon p = {r['w_p']:.4g} {stars(r['w_p'])}   |   "
            f"paired t p = {r['t_p']:.4g} {stars(r['t_p'])}")
        add(f"- effect size: Cohen's d_z = {fmt_d(r['dz'])}, "
            f"rank-biserial r = {r['rrb']:+.2f}\n")
        return r

    # --- 1. Headline: PPO vs A2C ------------------------------------------
    add("## 1. PPO vs A2C (headline)\n")
    a, b = paired_frame(df, "algo", "ppo", "a2c",
                        ["base_reward", "sin_cos", "robust", "seed"])
    report("PPO vs A2C (all 36 matched config-seed pairs)", a, b, "PPO", "A2C")

    # also the simple unpaired test (what a reader expects to see once)
    ppo, a2c = df[df.algo == "ppo"]["mean_reward"], df[df.algo == "a2c"]["mean_reward"]
    mw = stats.mannwhitneyu(ppo, a2c, alternative="two-sided")
    add(f"- _Unpaired cross-check:_ Mann-Whitney U p = {mw.pvalue:.3g} "
        f"{stars(mw.pvalue)} (PPO median {ppo.median():.1f} vs "
        f"A2C median {a2c.median():.1f}).\n")

    # --- 2. Factor effects within each algorithm --------------------------
    add("## 2. Design-factor effects (within each algorithm, paired)\n")
    for algo in ("ppo", "a2c"):
        sub = df[df.algo == algo]
        add(f"### {algo.upper()}")
        # reward shaping: pair over (sin_cos, robust, seed)
        a, b = paired_frame(sub, "base_reward", True, False,
                            ["sin_cos", "robust", "seed"])
        report(f"{algo.upper()}: base reward vs tuned reward",
               a, b, "base", "tuned")
        # observation: raw exists only at nominal -> restrict robust=False
        nominal = sub[sub.robust == False]
        a, b = paired_frame(nominal, "sin_cos", True, False,
                            ["base_reward", "seed"])
        report(f"{algo.upper()}: sin/cos vs raw obs (nominal only)",
               a, b, "sin/cos", "raw")
        # robustness: robust exists only with sin_cos=True
        sincos = sub[sub.sin_cos == True]
        a, b = paired_frame(sincos, "robust", True, False,
                            ["base_reward", "seed"])
        report(f"{algo.upper()}: robust vs nominal training (sin/cos only)",
               a, b, "robust", "nominal")

    # --- 3. Controller selection on the RIGHT metric (steps-to-threshold) --
    add("## 3. Controller selection among the configs that reach the cap\n")
    add("Final return cannot separate the censored configs (all ~180). The "
        "fair questions are: *which configs reliably reach the cap*, and "
        "*how fast*. Reliability = fraction of seeds hitting 180; speed = "
        "steps-to-threshold (lower is better), compared paired by seed.\n")

    capable = (df[(df.algo == "ppo") & df.censored]
               .groupby(["base_reward", "sin_cos", "robust"]))
    summary = capable.agg(hit_rate=("seed", "size"),
                          steps_mean=("num_timesteps", "mean"),
                          time_mean=("train_time_min", "mean"))
    # hit_rate currently = #censored seeds; express as out of 6
    summary["hit_rate"] = summary["hit_rate"].astype(int)
    add("```")
    add(summary.round(0).astype({"steps_mean": int}).to_string())
    add("```")
    add("(hit_rate = seeds out of 6 that reached 180)\n")

    # fastest reliable (6/6) config = reference; compare on steps-to-threshold
    reliable = summary[summary.hit_rate == 6].sort_values("steps_mean")
    if len(reliable) >= 2:
        best_key = reliable.index[0]
        add(f"Fastest 6/6-reliable config = base_reward={best_key[0]}, "
            f"sin_cos={best_key[1]}, robust={best_key[2]} "
            f"({int(reliable.iloc[0]['steps_mean'])} steps). "
            "Steps-to-threshold vs the other reliable configs, paired by seed:\n")

        def cfg_steps(key):
            m = df[(df.algo == "ppo") & df.censored
                   & (df.base_reward == key[0]) & (df.sin_cos == key[1])
                   & (df.robust == key[2])]
            return m.set_index("seed")["num_timesteps"].sort_index()

        ref = cfg_steps(best_key)
        for key in reliable.index[1:]:
            other = cfg_steps(key)
            common = ref.index.intersection(other.index)
            r = paired_test(ref.loc[common].values.astype(float),
                            other.loc[common].values.astype(float))
            label = f"baseR={key[0]},sincos={key[1]},robust={key[2]}"
            add(f"- vs `{label}`: {-r['mean_diff']:+.0f} more steps for it "
                f"(i.e. reference is faster by {-r['mean_diff']:.0f}), "
                f"Wilcoxon p={r['w_p']:.3g} {stars(r['w_p'])}")
    add("\nRobust training keeps the cap reachable but is **not free**: it "
        "needs more steps to get there, and with tuned reward only 4/6 seeds "
        "reached it at all - the robustness/sample-cost trade-off the final "
        "return hides.\n")

    # --- 4. Variance / stability across seeds -----------------------------
    add("## 4. Seed stability (variance, not just mean)\n")
    lev = stats.levene(ppo.values, a2c.values, center="median")
    add(f"- Levene test of equal variance PPO vs A2C: "
        f"p = {lev.pvalue:.3g} {stars(lev.pvalue)} "
        f"(PPO std {ppo.std():.1f} vs A2C std {a2c.std():.1f}). "
        "A2C is not just lower but markedly less reproducible across seeds.\n")

    # per-config coefficient of variation
    cv = (df.groupby(["algo", "base_reward", "sin_cos", "robust"])["mean_reward"]
          .agg(["mean", "std"]))
    cv["cv_%"] = 100 * cv["std"] / cv["mean"]
    add("Per-config seed spread (coefficient of variation):\n")
    add("```")
    add(cv.round(1).to_string())
    add("```\n")

    # --- 5. Forest plot ---------------------------------------------------
    plot_forest(rows_for_plot)

    add("## Bottom line\n")
    add("- PPO beats A2C with a large, significant paired effect and far "
        "lower seed variance; the algorithm choice is statistically decisive. "
        "(PPO/sincos is censored at 180, so the true gap is even larger.)")
    add("- sin/cos obs is the one design factor with a real, significant "
        "return gain over raw - and since sincos is censored, that gain is a "
        "lower bound.")
    add("- Robust training does **not** lower final return, but it is **not "
        "free**: it costs extra steps-to-threshold and, with tuned reward, "
        "costs reliability (4/6 vs 6/6 seeds reaching the cap). The trade is "
        "sample-cost for sim-to-real robustness, not zero-cost insurance.")
    add("- The top PPO configs are tied on reward only because they are all "
        "censored at 180; on steps-to-threshold the controller "
        "can be chosen on cost/robustness rather than raw return.")

    emit_latex(df, test_rows)

    OUT_PATH.write_text("\n".join(lines))
    print(f"\nWrote {OUT_PATH.relative_to(HERE)}")


def emit_latex(df, test_rows, path=HERE / "tables_exp1.tex"):
    """Write the two data-driven LaTeX tables consumed by results_exp1.tex."""
    def p_tex(p):
        s = stars(p)
        s = "" if s == "ns" else f"\\textsuperscript{{{s}}}"
        return ("$<$0.001" if p < 1e-3 else f"{p:.3f}") + s

    # ---- Table 1: per-config summary (6 seeds) --------------------------
    g = (df.groupby(["algo", "base_reward", "sin_cos", "robust"])
         .agg(rmean=("mean_reward", "mean"), rstd=("mean_reward", "std"),
              steps=("num_timesteps", "mean"), time=("train_time_min", "mean"),
              hit=("censored", "sum")).reset_index())
    g["rstd"] = g["rstd"].fillna(0.0)
    rew = {True: "base", False: "tuned"}
    obs = {True: "sincos", False: "raw"}
    dis = {True: "rob.", False: "nom."}

    def cfg_rows(algo):
        sub = g[g.algo == algo].sort_values("rmean", ascending=False)
        out = []
        for _, r in sub.iterrows():
            ret = f"{r.rmean:.1f} $\\pm$ {r.rstd:.1f}"
            if r.hit >= 1:  # at least one seed censored at the cap
                ret = f"\\textbf{{{ret}}}"
            out.append(f"{algo.upper()} & {rew[r.base_reward]} & "
                       f"{obs[r.sin_cos]} & {dis[r.robust]} & "
                       f"{r.steps/1000:.0f}k & {r.time:.1f} & {ret} & "
                       f"{int(r.hit)}/6 \\\\")
        return out

    t1 = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Swing-up results over 6 seeds (42--47) per configuration. "
        r"``Return'' is mean $\pm$ std over seeds of the mean return over 10 "
        r"eval episodes. ``$n_{180}$'' is the number of seeds that reached the "
        r"early-stopping cap of 180; \textbf{bold} return marks censored "
        r"configurations (return is then a floor, not a ceiling).}",
        r"\label{tab:exp1}",
        r"\begin{tabular}{llll rr r c}", r"\toprule",
        r"Algo & Reward & Obs & Dist. & Steps & Time [min] & Return & "
        r"$n_{180}$ \\", r"\midrule",
        *cfg_rows("ppo"), r"\midrule", *cfg_rows("a2c"),
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]

    # ---- Table 2: key paired tests -------------------------------------
    def trow(r):
        return (f"{r['title'].replace('&', r'\&')} & {r['n']} & "
                f"${r['mean_diff']:+.1f}$ & "
                f"$[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]$ & "
                f"${r['dz']:+.2f}$ & {p_tex(r['w_p'])} \\\\")

    t2 = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Paired comparisons (within-seed blocks). $\Delta$ is the "
        r"mean paired difference in return; CI is a 95\% percentile bootstrap; "
        r"$d_z$ is the paired effect size; $p$ is the paired Wilcoxon "
        r"signed-rank test (\textsuperscript{*}$<$0.05, "
        r"\textsuperscript{**}$<$0.01, \textsuperscript{***}$<$0.001). "
        r"Comparisons against a censored side (sin/cos, PPO) are conservative.}",
        r"\label{tab:exp1_tests}",
        r"\begin{tabular}{l c r c c c}", r"\toprule",
        r"Comparison & $n$ & $\Delta$ & 95\% CI & $d_z$ & Wilcoxon $p$ \\",
        r"\midrule",
        *[trow(r) for r in test_rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]

    path.write_text("\n".join(t1 + t2))
    print(f"  saved {path.relative_to(HERE)}")


def plot_forest(rows):
    rows = [r for r in rows if np.isfinite(r[1])]
    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    los = np.array([r[2] for r in rows])
    his = np.array([r[3] for r in rows])
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.5))
    err = np.vstack([means - los, his - means])
    colors = ["#2ca02c" if lo > 0 else "#d62728" if hi < 0 else "#7f7f7f"
              for lo, hi in zip(los, his)]
    ax.errorbar(means, y, xerr=err, fmt="none", ecolor="k", capsize=3, zorder=1)
    ax.scatter(means, y, c=colors, s=45, zorder=2)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mean paired difference in return (95% bootstrap CI)")
    ax.set_title("Effect sizes: green = significant gain, red = loss, grey = tie")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "effect_sizes.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.relative_to(HERE)}")


if __name__ == "__main__":
    main()
