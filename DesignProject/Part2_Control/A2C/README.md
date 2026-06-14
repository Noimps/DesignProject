# Actor-Critic (A2C) Swing-Up Controller

This folder contains an Advantage Actor-Critic (A2C) agent that learns to **swing up
and balance** the *unbalanced disk* to its upright position (`theta = pi`).

- `actor_critic_swingup.py` — the full training + evaluation script.
- `actor_critic_outputs_motion_reward_long/` — example outputs from a finished run
  (saved checkpoints and PNG plots).

---

## What the script does

### Environment
It uses the [`gym_unbalanced_disk`](https://pypi.org/project/gym-unbalanced-disk/)
Gymnasium environment. The default env id is `unbalanced-disk-v0`, with:

- `dt = 0.025` s simulation step,
- `umax = 3.0` V motor voltage limit (the action).

The raw observation can be `[theta, omega]` or a `sin/cos` encoding; the script handles
both via `--obs-type` and the `extract_theta_omega` helper.

### State representation
Regardless of the raw observation, the policy always sees a 3-D state:

```
state = [sin(theta), cos(theta), omega]
```

This avoids the angle wrap-around discontinuity at `±pi`.

### Reward
A custom **swing-up reward** (`swingup_reward`) is used by default. It is shaped to
reward being upright with low velocity and low control effort:

```
upright_score = exp(-0.5 * ((theta_err / 0.7)^2 + (omega / 4.0)^2))
reward = 2.0 * upright_score - 1.0 + 0.03*|omega| - 0.002*omega^2 - 0.0005*u^2
```

where `theta_err = wrap_to_pi(theta - pi)`. The small `+0.03*|omega|` term encourages
the disk to build momentum (needed to swing up) while the quadratic terms penalize
excessive speed and voltage. Pass `--use-env-reward` to use the simulator's built-in
reward instead.

### Network (`ActorCritic`)
A shared MLP trunk feeds two heads:

- **Actor** — outputs the mean of a Gaussian over the voltage, squashed with
  `umax * tanh(...)`. A learned `log_std` parameter controls exploration noise.
- **Critic** — estimates the state value `V(s)`.

Default trunk: 2 hidden layers of 128 units with `Tanh` activations.

### Training loop
1. Collect a rollout of `--rollout-steps` (default 2048) environment steps using the
   current stochastic policy. Episodes are capped at `--episode-steps` (default 800).
2. Compute discounted returns (bootstrapped with the critic at the end of the rollout)
   and **normalized advantages**.
3. Optimize a combined loss:
   `actor_loss + value_coef * critic_loss - entropy_coef * entropy`,
   with gradient clipping (`--max-grad-norm`).
4. Repeat until `--total-steps` (default 200,000) environment steps are reached.

Checkpoints (`actor_critic_latest.pt`, and `actor_critic_final.pt` at the end) and
training-curve plots are written to `--out-dir`.

### Evaluation
`--mode eval` loads a checkpoint and runs the policy for `--eval-steps` (default 1000)
steps, printing summary stats (mean reward, mean angle error, mean |omega|, mean
|action|) and saving trajectory plots (theta, theta error, omega, action, reward).

---

## Requirements

```bash
pip install torch numpy matplotlib gymnasium gym-unbalanced-disk
```

A GPU is optional — CPU training is fine for this small network.

---

## How to run

Run from inside this folder (`Part2_Control/A2C/`).

### Train (defaults)
```bash
python actor_critic_swingup.py --mode train
```

### Train with common overrides
```bash
python actor_critic_swingup.py --mode train \
    --total-steps 300000 \
    --rollout-steps 2048 \
    --lr 3e-4 \
    --obs-type theta_omega \
    --out-dir my_run
```

### Evaluate a trained model
```bash
python actor_critic_swingup.py --mode eval \
    --model-path actor_critic_outputs_motion_reward_long/actor_critic_final.pt \
    --eval-steps 1000 \
    --render
```

> Note: `--out-dir` and `--model-path` default to a `SystemModelling/...` path, so pass
> them explicitly to read/write inside this folder (e.g. the included
> `actor_critic_outputs_motion_reward_long/`).

---

## Key command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | `train` or `eval`. |
| `--env-id` | `unbalanced-disk-v0` | Gym environment id. |
| `--obs-type` | `theta_omega` | Raw obs format: `auto`, `theta_omega`, `sin_cos_omega`, `cos_sin_omega`. |
| `--dt` | `0.025` | Simulation timestep (s). |
| `--umax` | `3.0` | Max voltage / action limit (V). |
| `--total-steps` | `200000` | Total env steps to train. |
| `--rollout-steps` | `2048` | Steps collected per update. |
| `--episode-steps` | `800` | Max steps per episode before reset. |
| `--gamma` | `0.99` | Discount factor. |
| `--lr` | `3e-4` | Adam learning rate. |
| `--hidden-dim` | `128` | Hidden units per layer. |
| `--value-coef` | `0.5` | Critic loss weight. |
| `--entropy-coef` | `0.01` | Entropy bonus weight (exploration). |
| `--max-grad-norm` | `0.5` | Gradient clipping norm. |
| `--seed` | `42` | RNG seed. |
| `--use-env-reward` | off | Use simulator reward instead of custom swing-up reward. |
| `--use-gpu` | off | Use CUDA if available. |
| `--out-dir` | `SystemModelling/actor_critic_outputs` | Where checkpoints/plots are saved. |
| `--log-interval` | `10000` | Console logging interval (steps). |
| `--save-interval` | `50000` | Checkpoint interval (steps). |
| `--model-path` | `.../actor_critic_final.pt` | Checkpoint to load for eval. |
| `--eval-steps` | `1000` | Steps to run during eval. |
| `--render` | off | Render the environment during eval. |

---

## Outputs

**Training** (in `--out-dir`):
- `actor_critic_latest.pt`, `actor_critic_final.pt` — checkpoints (model weights, args,
  and full training logs).
- `training_mean_reward.png`, `training_losses.png`, `training_entropy.png`,
  `training_log_std.png`.

**Evaluation** (in `--out-dir`):
- `eval_theta.png`, `eval_theta_error.png`, `eval_omega.png`, `eval_action.png`,
  `eval_reward.png`.
