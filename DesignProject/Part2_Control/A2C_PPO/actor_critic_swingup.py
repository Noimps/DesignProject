from pathlib import Path
import argparse
import sys
import time
import math
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


import gymnasium as gym


def _ensure_gym_unbalanced_disk_on_path():
    """
    The gym_unbalanced_disk package ships inside this repository
    (<repo>/gym-unbalanced-disk). If it has not been pip-installed, add that
    folder to sys.path so the import (and its env registration) still works.
    """
    repo_root = Path(__file__).resolve().parents[3]
    pkg_dir = repo_root / "gym-unbalanced-disk"
    if pkg_dir.is_dir() and str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))


# Importing gym_unbalanced_disk has the side effect of registering the
# 'unbalanced-disk-*' environments with gymnasium, so it must happen before
# any gym.make(...) call.
try:
    import gym_unbalanced_disk  # noqa: F401  (imported for registration side effect)
except ImportError:
    _ensure_gym_unbalanced_disk_on_path()
    try:
        import gym_unbalanced_disk  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Could not import gym_unbalanced_disk. Install it with "
            "`pip install -e gym-unbalanced-disk` from the repository root, "
            "or make sure the 'gym-unbalanced-disk' folder is on PYTHONPATH.\n"
        ) from e



def make_env(env_id, dt=0.025, umax=3.0):
    try:
        env = gym.make(
            env_id,
            dt=dt,
            umax=umax,
            disable_env_checker=True
        )
    except TypeError:
        env = gym.make(
            env_id,
            disable_env_checker=True
        )

    return env.unwrapped


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wrap_to_pi(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def safe_reset(env):
    """
    Handle old and new reset APIs.
    """
    out = env.reset()

    if isinstance(out, tuple):
        obs, info = out
    else:
        obs = out

    return np.asarray(obs, dtype=np.float32)


def safe_step(env, action):
    """
    The unbalanced-disk environment expects a scalar voltage input, not an
    array, so convert to a scalar float before env.step(...).

    The action is clipped to the environment's configured voltage limit
    (env.umax) rather than a hard-coded value, so it respects --umax.

    Returns (obs, reward, terminated, truncated, info), keeping termination and
    truncation separate (handling both the new 5-tuple and old 4-tuple APIs).
    """
    action = np.asarray(action).reshape(-1)
    action_scalar = float(action[0])

    limit = float(getattr(env, "umax", 3.0))
    action_scalar = float(np.clip(action_scalar, -limit, limit))

    out = env.step(action_scalar)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
    else:
        obs, reward, done, info = out
        terminated, truncated = bool(done), False

    return (
        np.asarray(obs, dtype=np.float32),
        float(reward),
        bool(terminated),
        bool(truncated),
        info,
    )


# Observation and reward utilities
def extract_theta_omega(obs, obs_type="auto"):
    """
    Extract theta and omega from the raw simulator observation.
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    if obs_type == "theta_omega":
        theta = float(obs[0])
        omega = float(obs[1]) if len(obs) > 1 else 0.0
        return theta, omega

    if obs_type == "sin_cos_omega":
        sin_theta = float(obs[0])
        cos_theta = float(obs[1])
        omega = float(obs[2]) if len(obs) > 2 else 0.0
        theta = math.atan2(sin_theta, cos_theta)
        return theta, omega

    if obs_type == "cos_sin_omega":
        cos_theta = float(obs[0])
        sin_theta = float(obs[1])
        omega = float(obs[2]) if len(obs) > 2 else 0.0
        theta = math.atan2(sin_theta, cos_theta)
        return theta, omega

    if obs_type == "auto":
        if len(obs) == 2:
            theta = float(obs[0])
            omega = float(obs[1])
            return theta, omega

        if len(obs) >= 3:
            sin_theta = float(obs[0])
            cos_theta = float(obs[1])
            omega = float(obs[2])
            theta = math.atan2(sin_theta, cos_theta)
            return theta, omega

    raise ValueError(f"Unsupported observation shape: {obs.shape}")


def preprocess_obs(obs, obs_type="auto"):
    """
    Convert raw observation into actor-critic state.
    """
    theta, omega = extract_theta_omega(obs, obs_type=obs_type)

    state = np.array(
        [
            math.sin(theta),
            math.cos(theta),
            omega,
        ],
        dtype=np.float32
    )

    return state


def swingup_reward(obs, action, obs_type="auto"):
    """
    Swing-up reward configuration.

    Reward is higher when:
        1 theta is close to pi
        2 angular velocity is small
        3 voltage effort is small
    """
    theta, omega = extract_theta_omega(obs, obs_type=obs_type)

    theta_error = wrap_to_pi(theta - np.pi)

    action = np.asarray(action).reshape(-1)
    u = float(action[0])

    # cost = (
    #     0.5 * theta_error ** 2
    #     + 0.01 * omega ** 2
    #     + 0.001 * u ** 2
    # )

    # reward = -cost

    # # Bonus for being close to upright and not moving too fast
    # if abs(theta_error) < 0.20 and abs(omega) < 1.0:
    #     reward += 2.0

    theta_error = wrap_to_pi(theta - np.pi)


    theta_scale = 0.7
    omega_scale = 4.0

    upright_score = math.exp(
        -0.5 * (
            (theta_error / theta_scale) ** 2
            + (omega / omega_scale) ** 2
        )
    )

    # reward = (
    #     2.0 * upright_score
    #     - 1.0
    #     - 0.001 * u ** 2
    # )

    reward = (
        2.0 * upright_score
        - 1.0
        + 0.03 * abs(omega)
        - 0.002 * omega ** 2
        - 0.0005 * u ** 2
    )
    return float(reward)





# Actor-Critic network
class ActorCritic(nn.Module):
    """
     actor-critic network.

    State:[sin(theta), cos(theta), omega]

    Actor:outputs a Gaussian policy over voltage action.

    Critic:estimates V.
    """

    def __init__(self, state_dim=3, hidden_dim=128, action_limit=3.0):
        super().__init__()

        self.action_limit = action_limit

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.critic_value = nn.Linear(hidden_dim, 1)

        # Learned log standard deviation for stochastic policy
        #self.log_std = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
        #self.log_std = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.log_std = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))


    def forward(self, state):
        z = self.shared(state)

        mean_raw = self.actor_mean(z)
        mean = self.action_limit * torch.tanh(mean_raw)

        value = self.critic_value(z).squeeze(-1)

        std = torch.exp(self.log_std).expand_as(mean)

        return mean, std, value

    def act(self, state, deterministic=False):
        mean, std, value = self.forward(state)

        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action).squeeze(-1)
        else:
            dist = Normal(mean, std)
            action = dist.rsample()
            action = torch.clamp(action, -self.action_limit, self.action_limit)
            log_prob = dist.log_prob(action).squeeze(-1)

        return action, log_prob, value


# Rollout collection

def collect_rollout(
    env,
    model,
    device,
    rollout_steps,
    gamma,
    obs_type,
    use_env_reward=False,
    episode_steps=800,
):
    """
    Collect one rollout for actor-critic training.

    """
    obs = safe_reset(env)
    state_np = preprocess_obs(obs, obs_type=obs_type)

    states = []
    actions = []
    log_probs = []
    rewards = []
    terminateds = []
    truncateds = []
    values = []
    next_values = []

    episode_rewards = []
    current_episode_reward = 0.0
    steps_in_episode = 0

    for _ in range(rollout_steps):
        state = torch.tensor(
            state_np,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        action_t, log_prob_t, value_t = model.act(state, deterministic=False)

        action_np = action_t.detach().cpu().numpy().reshape(-1)

        next_obs, env_reward, terminated, truncated, info = safe_step(env, action_np)

        if use_env_reward:
            reward = env_reward
        else:
            reward = swingup_reward(next_obs, action_np, obs_type=obs_type)

        next_state_np = preprocess_obs(next_obs, obs_type=obs_type)

        current_episode_reward += reward
        steps_in_episode += 1

        # Reaching the episode-step cap is a time-limit *truncation*, not a true
        # terminal state: the real dynamics would continue, so the return must
        # bootstrap with V(next_state) rather than being cut off at zero.
        if steps_in_episode >= episode_steps:
            truncated = True

        done = terminated or truncated

        # For a truncated (but not terminated) step we need the value of the
        # state we are about to leave behind, so its return can be bootstrapped.
        if truncated and not terminated:
            with torch.no_grad():
                next_state = torch.tensor(
                    next_state_np,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)
                _, _, nv = model.forward(next_state)
                next_value_t = nv.squeeze(0)
        else:
            next_value_t = torch.zeros((), dtype=torch.float32, device=device)

        states.append(state.squeeze(0))
        actions.append(action_t.squeeze(0))
        log_probs.append(log_prob_t.squeeze(0))
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        terminateds.append(torch.tensor(float(terminated), dtype=torch.float32, device=device))
        truncateds.append(torch.tensor(float(truncated), dtype=torch.float32, device=device))
        values.append(value_t.squeeze(0))
        next_values.append(next_value_t)

        if done:
            episode_rewards.append(current_episode_reward)

            current_episode_reward = 0.0
            steps_in_episode = 0

            obs = safe_reset(env)
            state_np = preprocess_obs(obs, obs_type=obs_type)
        else:
            obs = next_obs
            state_np = next_state_np

    # Bootstrap value for the (possibly unfinished) final transition. This is
    # only used for trailing steps that did not end an episode; terminated and
    # truncated steps override it in the recursion below.
    state = torch.tensor(
        state_np,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        _, _, bootstrap_value = model.forward(state)
        bootstrap_value = bootstrap_value.squeeze(0)

    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    terminateds = torch.stack(terminateds)
    truncateds = torch.stack(truncateds)
    values = torch.stack(values)
    next_values = torch.stack(next_values)

    returns = []
    G = bootstrap_value

    for t in reversed(range(rollout_steps)):
        # terminated -> no bootstrap          (G = reward)
        # truncated  -> bootstrap V(next_state), do not carry the future G
        # otherwise  -> standard discounted accumulation
        bootstrap = truncateds[t] * next_values[t] + (1.0 - truncateds[t]) * G
        G = rewards[t] + gamma * bootstrap * (1.0 - terminateds[t])
        returns.insert(0, G)

    returns = torch.stack(returns)

    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rollout = {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "rewards": rewards,
        "terminateds": terminateds,
        "truncateds": truncateds,
        "values": values,
        "returns": returns.detach(),
        "advantages": advantages.detach(),
        "episode_rewards": episode_rewards,
    }

    return rollout


# Training

def train_actor_critic(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )

    print(f"Using device: {device}")

    env = make_env(args.env_id, dt=args.dt, umax=args.umax)

    model = ActorCritic(
        state_dim=3,
        hidden_dim=args.hidden_dim,
        action_limit=args.umax,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    all_episode_rewards = []
    mean_reward_log = []
    actor_loss_log = []
    critic_loss_log = []
    entropy_log = []
    log_std_log = []

    print("\nStarting actor-critic training")
    print(f"Environment: {args.env_id}")
    print(f"Observation type: {args.obs_type}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Episode steps: {args.episode_steps}")
    print(f"Total training steps: {args.total_steps}")
    print(f"Using custom reward: {not args.use_env_reward}")

    t0 = time.time()

    while total_steps < args.total_steps:
        rollout = collect_rollout(
            env=env,
            model=model,
            device=device,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            obs_type=args.obs_type,
            use_env_reward=args.use_env_reward,
            episode_steps=args.episode_steps,
        )

        states = rollout["states"]
        actions = rollout["actions"]
        returns = rollout["returns"]
        advantages = rollout["advantages"]

        mean, std, values = model.forward(states)
        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).squeeze(-1)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)

        loss = (
            actor_loss
            + args.value_coef * critic_loss
            - args.entropy_coef * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        total_steps += args.rollout_steps

        if rollout["episode_rewards"]:
            all_episode_rewards.extend(rollout["episode_rewards"])

        if len(all_episode_rewards) >= 10:
            mean_last = float(np.mean(all_episode_rewards[-10:]))
        elif len(all_episode_rewards) > 0:
            mean_last = float(np.mean(all_episode_rewards))
        else:
            mean_last = np.nan

        mean_reward_log.append(mean_last)
        actor_loss_log.append(float(actor_loss.detach().cpu()))
        critic_loss_log.append(float(critic_loss.detach().cpu()))
        entropy_log.append(float(entropy.detach().cpu()))
        log_std_log.append(float(model.log_std.detach().cpu()))

        if total_steps % args.log_interval < args.rollout_steps:
            print(
                f"steps={total_steps:7d} | "
                f"mean_reward_10ep={mean_last:10.3f} | "
                f"actor_loss={actor_loss.item():10.4f} | "
                f"critic_loss={critic_loss.item():10.4f} | "
                f"entropy={entropy.item():8.4f} | "
                f"log_std={model.log_std.item():8.4f}"
            )

        if total_steps % args.save_interval < args.rollout_steps:
            save_path = out_dir / "actor_critic_latest.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "total_steps": total_steps,
                    "episode_rewards": all_episode_rewards,
                    "mean_reward_log": mean_reward_log,
                    "actor_loss_log": actor_loss_log,
                    "critic_loss_log": critic_loss_log,
                    "entropy_log": entropy_log,
                    "log_std_log": log_std_log,
                },
                save_path,
            )
            print(f"Saved latest checkpoint: {save_path}")

    env.close()

    final_path = out_dir / "actor_critic_final.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "total_steps": total_steps,
            "episode_rewards": all_episode_rewards,
            "mean_reward_log": mean_reward_log,
            "actor_loss_log": actor_loss_log,
            "critic_loss_log": critic_loss_log,
            "entropy_log": entropy_log,
            "log_std_log": log_std_log,
        },
        final_path,
    )

    print("\nTraining finished.")
    print(f"Training time: {time.time() - t0:.1f} s")
    print(f"Saved final model: {final_path}")

    plot_training_curves(
        out_dir=out_dir,
        mean_reward_log=mean_reward_log,
        actor_loss_log=actor_loss_log,
        critic_loss_log=critic_loss_log,
        entropy_log=entropy_log,
        log_std_log=log_std_log,
    )

    return model


# Evaluation

def evaluate_policy(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    saved_args = checkpoint["args"]

    hidden_dim = saved_args.get("hidden_dim", args.hidden_dim)
    umax = saved_args.get("umax", args.umax)

    model = ActorCritic(
        state_dim=3,
        hidden_dim=hidden_dim,
        action_limit=umax,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    env = make_env(args.env_id, dt=args.dt, umax=args.umax)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = safe_reset(env)
    state_np = preprocess_obs(obs, obs_type=args.obs_type)

    theta_list = []
    omega_list = []
    action_list = []
    reward_list = []

    total_reward = 0.0

    for t in range(args.eval_steps):
        theta, omega = extract_theta_omega(obs, obs_type=args.obs_type)

        state = torch.tensor(
            state_np,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():
            action_t, _, _ = model.act(state, deterministic=False)

        action_np = action_t.cpu().numpy().reshape(-1)
        action_np = np.clip(action_np, -args.umax, args.umax)

        next_obs, env_reward, terminated, truncated, info = safe_step(env, action_np)
        done = terminated or truncated

        if args.use_env_reward:
            reward = env_reward
        else:
            reward = swingup_reward(next_obs, action_np, obs_type=args.obs_type)

        theta_list.append(theta)
        omega_list.append(omega)
        action_list.append(float(action_np[0]))
        reward_list.append(reward)

        total_reward += reward

        if args.render:
            env.render()

        if done:
            obs = safe_reset(env)
        else:
            obs = next_obs

        state_np = preprocess_obs(obs, obs_type=args.obs_type)

    env.close()

    theta_arr = np.asarray(theta_list)
    omega_arr = np.asarray(omega_list)
    action_arr = np.asarray(action_list)
    reward_arr = np.asarray(reward_list)

    theta_error = wrap_to_pi(theta_arr - np.pi)

    print("\nEvaluation")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Mean reward : {np.mean(reward_arr):.3f}")
    print(f"Mean |theta error to upright|: {np.mean(np.abs(theta_error)):.3f} rad")
    print(f"Mean |omega|: {np.mean(np.abs(omega_arr)):.3f} rad/s")
    print(f"Mean |action|: {np.mean(np.abs(action_arr)):.3f} V")

    plot_evaluation(
        out_dir=out_dir,
        theta=theta_arr,
        omega=omega_arr,
        action=action_arr,
        reward=reward_arr,
        theta_error=theta_error,
    )


# Plotting


def plot_training_curves(
    out_dir,
    mean_reward_log,
    actor_loss_log,
    critic_loss_log,
    entropy_log,
    log_std_log,
):
    out_dir = Path(out_dir)

    plt.figure(figsize=(10, 4))
    plt.plot(mean_reward_log)
    plt.xlabel("Update")
    plt.ylabel("Mean reward over last episodes")
    plt.title("Actor-Critic training reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "training_mean_reward.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(actor_loss_log, label="Actor loss")
    plt.plot(critic_loss_log, label="Critic loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title("Actor and critic losses")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_losses.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(entropy_log)
    plt.xlabel("Update")
    plt.ylabel("Entropy")
    plt.title("Policy entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "training_entropy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(log_std_log)
    plt.xlabel("Update")
    plt.ylabel("log_std")
    plt.title("Policy log standard deviation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "training_log_std.png", dpi=200)
    plt.close()


def plot_evaluation(out_dir, theta, omega, action, reward, theta_error):
    out_dir = Path(out_dir)
    t = np.arange(len(theta))

    plt.figure(figsize=(12, 4))
    plt.plot(t, theta, label="theta")
    plt.axhline(np.pi, linestyle="--", label="upright target pi")
    plt.axhline(-np.pi, linestyle="--", label="-pi")
    plt.xlabel("Step")
    plt.ylabel("Angle [rad]")
    plt.title("Evaluation angle trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eval_theta.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t, theta_error)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("theta - pi wrapped [rad]")
    plt.title("Evaluation angle error to upright")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "eval_theta_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t, omega)
    plt.xlabel("Step")
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Evaluation angular velocity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "eval_omega.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t, action)
    plt.axhline(3.0, linestyle="--")
    plt.axhline(-3.0, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Voltage action [V]")
    plt.title("Evaluation action signal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "eval_action.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t, reward)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Evaluation reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "eval_reward.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
    )

    parser.add_argument(
        "--env-id",
        type=str,
        default="unbalanced-disk-v0",
        help="Environment ID. Usually unbalanced-disk-v0 or unbalanced-disk-sincos-v0.",
    )

    parser.add_argument(
        "--obs-type",
        type=str,
        default="theta_omega",
        choices=["auto", "theta_omega", "sin_cos_omega", "cos_sin_omega"],
    )

    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--umax", type=float, default=3.0)

    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--episode-steps", type=int, default=800)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use-env-reward",
        action="store_true",
        help="Use simulator reward instead of custom swing-up reward.",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available. CPU is fine.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="SystemModelling/actor_critic_outputs",
    )

    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--save-interval", type=int, default=50_000)

    parser.add_argument(
        "--model-path",
        type=str,
        default="SystemModelling/actor_critic_outputs/actor_critic_final.pt",
    )

    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    if args.mode == "train":
        train_actor_critic(args)
    else:
        evaluate_policy(args)


if __name__ == "__main__":
    main()