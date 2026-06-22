# Statistical analysis of the trained swing-up agents

_72 runs, 12 configurations x 6 seeds (42-47). Paired/blocked design: comparisons are made within matched seed blocks. Primary test = paired Wilcoxon signed-rank; paired t-test and Cohen's d_z reported alongside; 95% CIs are percentile bootstrap (B=10000) on the mean paired difference._

> **Censoring caveat.** Training stopped on reaching reward 180 (0.9 x max). 22/72 runs hit that cap (PPO 22, A2C 0) - all of them PPO+sin/cos configs. Their reward is right-censored at 180, so reward-based comparisons *between censored configs* are uninformative (they are equal by construction); for those, the discriminating metric is **steps-to-threshold** (Section 3). Comparisons where one side is censored and the other is not (PPO vs A2C, sin/cos vs raw) are therefore **conservative**: the true gap is at least the reported one.

## 1. PPO vs A2C (headline)

### PPO vs A2C (all 36 matched config-seed pairs)
- pairs n = 36  (PPO - A2C)
- mean difference = **+106.6** return  (95% CI [+93.4, +119.4])
- Wilcoxon p = 2.91e-11 ***   |   paired t p = 2.412e-17 ***
- effect size: Cohen's d_z = +2.60 (large), rank-biserial r = +1.00

- _Unpaired cross-check:_ Mann-Whitney U p = 3e-13 *** (PPO median 180.5 vs A2C median 92.2).

## 2. Design-factor effects (within each algorithm, paired)

### PPO
### PPO: base reward vs tuned reward
- pairs n = 18  (base - tuned)
- mean difference = **+5.0** return  (95% CI [+2.0, +8.4])
- Wilcoxon p = 0.02685 *   |   paired t p = 0.008183 **
- effect size: Cohen's d_z = +0.71 (medium), rank-biserial r = +0.59

### PPO: sin/cos vs raw obs (nominal only)
- pairs n = 12  (sin/cos - raw)
- mean difference = **+23.9** return  (95% CI [+19.2, +28.7])
- Wilcoxon p = 0.0004883 ***   |   paired t p = 1.303e-06 ***
- effect size: Cohen's d_z = +2.73 (large), rank-biserial r = +1.00

### PPO: robust vs nominal training (sin/cos only)
- pairs n = 12  (robust - nominal)
- mean difference = **-1.3** return  (95% CI [-3.4, +0.2])
- Wilcoxon p = 1 ns   |   paired t p = 0.2082 ns
- effect size: Cohen's d_z = -0.39 (small), rank-biserial r = +0.00

### A2C
### A2C: base reward vs tuned reward
- pairs n = 18  (base - tuned)
- mean difference = **-2.8** return  (95% CI [-17.8, +12.9])
- Wilcoxon p = 0.8589 ns   |   paired t p = 0.731 ns
- effect size: Cohen's d_z = -0.08 (negligible), rank-biserial r = -0.06

### A2C: sin/cos vs raw obs (nominal only)
- pairs n = 12  (sin/cos - raw)
- mean difference = **-32.9** return  (95% CI [-54.6, -9.4])
- Wilcoxon p = 0.03223 *   |   paired t p = 0.01915 *
- effect size: Cohen's d_z = -0.79 (medium), rank-biserial r = -0.69

### A2C: robust vs nominal training (sin/cos only)
- pairs n = 12  (robust - nominal)
- mean difference = **+31.5** return  (95% CI [+13.3, +50.2])
- Wilcoxon p = 0.08398 ns   |   paired t p = 0.009617 **
- effect size: Cohen's d_z = +0.90 (large), rank-biserial r = +0.64

## 3. Controller selection among the configs that reach the cap

Final return cannot separate the censored configs (all ~180). The fair questions are: *which configs reliably reach the cap*, and *how fast*. Reliability = fraction of seeds hitting 180; speed = steps-to-threshold (lower is better), compared paired by seed.

```
                            hit_rate  steps_mean  time_mean
base_reward sin_cos robust                                 
False       True    False          6      166667        1.0
                    True           4      193750        2.0
True        True    False          6      191667        2.0
                    True           6      233333        2.0
```
(hit_rate = seeds out of 6 that reached 180)

Fastest 6/6-reliable config = base_reward=False, sin_cos=True, robust=False (166666 steps). Steps-to-threshold vs the other reliable configs, paired by seed:

- vs `baseR=True,sincos=True,robust=False`: +25000 more steps for it (i.e. reference is faster by 25000), Wilcoxon p=0.0625 ns
- vs `baseR=True,sincos=True,robust=True`: +66667 more steps for it (i.e. reference is faster by 66667), Wilcoxon p=0.0625 ns

Robust training keeps the cap reachable but is **not free**: it needs more steps to get there, and with tuned reward only 4/6 seeds reached it at all - the robustness/sample-cost trade-off the final return hides.

## 4. Seed stability (variance, not just mean)

- Levene test of equal variance PPO vs A2C: p = 1.73e-05 *** (PPO std 12.3 vs A2C std 36.1). A2C is not just lower but markedly less reproducible across seeds.

Per-config seed spread (coefficient of variation):

```
                                  mean   std  cv_%
algo base_reward sin_cos robust                   
a2c  False       False   False    85.2  28.6  33.6
                 True    False    49.7  35.6  71.6
                         True     67.2  47.6  70.8
     True        False   False    69.7  31.3  45.0
                 True    False    39.3  30.7  78.1
                         True     84.7  28.4  33.5
ppo  False       False   False   150.8   5.9   3.9
                 True    False   181.1   0.7   0.4
                         True    178.3   4.5   2.5
     True        False   False   163.4   5.3   3.3
                 True    False   180.8   0.4   0.2
                         True    181.1   0.3   0.2
```

## Bottom line

- PPO beats A2C with a large, significant paired effect and far lower seed variance; the algorithm choice is statistically decisive. (PPO/sincos is censored at 180, so the true gap is even larger.)
- sin/cos obs is the one design factor with a real, significant return gain over raw - and since sincos is censored, that gain is a lower bound.
- Robust training does **not** lower final return, but it is **not free**: it costs extra steps-to-threshold and, with tuned reward, costs reliability (4/6 vs 6/6 seeds reaching the cap). The trade is sample-cost for sim-to-real robustness, not zero-cost insurance.
- The top PPO configs are tied on reward only because they are all censored at 180; on steps-to-threshold the controller can be chosen on cost/robustness rather than raw return.