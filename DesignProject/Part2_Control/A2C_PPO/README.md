# A2C / PPO on the Unbalanced Disk

`P2_A2C_PPO.ipynb` trains and evaluates A2C and PPO agents (stable-baselines3)
on the unbalanced-disk swing-up task for the Advanced RL chapter of the report.

## Requirements
- `stable_baselines3`, `gymnasium`, `torch`, `numpy`, `matplotlib`, `joblib`, `filelock`
- `gym_unbalanced_disk` (registers the env ids; must be importable)

## Environment configs
Each run sweeps three flags:
- `base_reward` — base reward (`True`) vs. tuned reward (`False`, the current default in `UnbalancedDisc.py`)
- `sin_cos` — wrap-invariant `[sin θ, cos θ, ω]` observation vs. raw `[θ, ω]`
- `robust` — adds actuator noise during training

## Notebook layout
0. **Imports** — set seeds, single-threaded BLAS/torch.
1. **Helpers** — `make_env`, training/eval/plot/visualize utilities, callbacks, CSV logging.
2. **Experiments** — A2C and PPO grid sweeps over configs × seeds. Training loops are guarded by `continue`; remove it to retrain. A parallel (loky) PPO sweep is provided.
3. **Reference tracking** — best config (PPO, base reward, sin-cos, robust) trained to hold a reference angle off upright (`ref_angle`, default 15°).

## Outputs
- Trained models: `models/grid_search/`, `models/reference_tracking/`
- Metrics: `training_results.csv` (one row per run; aggregate with `analyze_results.py`)
- TensorBoard: `tensorboard --logdir ./tensorboard_logs`

## Visualizing a trained model
Set `model_name` (e.g. `ppo_tunedR_sincos_robust`) in section 2.2 and uncomment
`evaluate_trained_policy` / `visualize_trained_policy` / `plot_trained_policy`.
