# Training-results analysis

_72 runs across 12 configurations, 6–6 seeds each (36 PPO + 36 A2C runs)._

## Overall
- **Best config:** `PPO | tuned | sin/cos | nominal` — return **181.1 ± 0.7** over 6 seeds, in 1.4 min avg.
- **Worst config:** `A2C | base | sin/cos | nominal` — 39.3 ± 30.7.
- **Censoring caveat:** training stopped at reward 180 (0.9×max), so 22/72 runs (all PPO·sin/cos) are right-censored at 180 — their return is a *floor*, not a ceiling. Reward cannot separate the capped configs; for those, steps-to-threshold is the real metric (see `statistical_analysis.md`).

## Algorithm: PPO vs A2C
- Return (all runs) — PPO **172.6 ± 12.3** vs A2C **66.0 ± 36.1**.
- Training time — PPO 3.0 min vs A2C 12.6 min.
- PPO's per-config std stays small (max 5.9); A2C's is large (max 47.6) — i.e. A2C results are seed-sensitive, which is exactly why multiple seeds were needed.

## Design factors (mean ± std over runs)
- **Reward shaping:**
  - PPO: base reward 175.1 $\pm$ 9.0; tuned reward 170.0 $\pm$ 14.6
  - A2C: base reward 64.6 $\pm$ 34.4; tuned reward 67.4 $\pm$ 38.8
- **Observation:**
  - PPO: raw 157.1 $\pm$ 8.5; sin/cos 180.3 $\pm$ 2.5
  - A2C: raw 77.4 $\pm$ 29.7; sin/cos 60.2 $\pm$ 38.2
- **Disturbance:**
  - PPO: nominal 169.0 $\pm$ 13.6; robust 179.7 $\pm$ 3.4
  - A2C: nominal 61.0 $\pm$ 34.7; robust 76.0 $\pm$ 38.5

## Efficiency
- Best return-per-minute config: `PPO | tuned | sin/cos | nominal` (181 return in 1.4 min).
- Return vs training time correlates r=-0.81 across runs; more compute did not buy more return (the long runs are the A2C failures). Early stopping halted converged runs before the 1M cap.

## Bottom line
- **Use PPO.** Higher return, faster, and low seed variance. A2C is both
  weaker and unstable across seeds on this task.
- The top PPO configs are tied on return only because they are all
  censored at the 180 cap; the recommended controller is therefore the
  one that reaches the cap fastest and most reliably
  (`PPO | tuned | sin/cos | nominal`).
- Robust-trained PPO reaches the same cap, so it costs no *final return*
  — but it is not free: it needs more steps to get there (and with the
  tuned reward only 4/6 seeds reach the cap), trading sample-cost for
  sim-to-real robustness. See `statistical_analysis.md` §3.
- Reporting mean ± std over 6–6 seeds removes
  the single-seed confound from the earlier write-up.
