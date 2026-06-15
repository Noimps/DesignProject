"""
Train an A2C agent on the unbalanced-disk swing-up task using stable-baselines3.

Run from anywhere (the gym_unbalanced_disk import registers the env id):
    python a2c.py

View the training logs with:
    tensorboard --logdir ./tensorboard_logs
"""

import time
from pathlib import Path

import numpy as np
import gymnasium as gym

# Importing gym_unbalanced_disk registers the 'unbalanced-disk-v0' env id with
# gymnasium, so it must be imported before any gym.make(...) call.
import gym_unbalanced_disk  # noqa: F401  (imported for its registration side effect)
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
)


# All artifacts (checkpoints, logs) are written next to this script, regardless
# of the directory the script is launched from. In a Jupyter notebook __file__
# is not defined, so fall back to the current working directory.
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
TB_LOG_DIR = HERE / "tensorboard_logs"
BEST_MODEL_DIR = HERE / "a2c_best"
EVAL_LOG_DIR = HERE / "a2c_eval_logs"
CHECKPOINT_DIR = HERE / "a2c_checkpoints"
FINAL_MODEL_PATH = HERE / "a2c_unbalanced_disk"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env():
    """
    Build a single training/eval environment.

    We use gym.make(...) with the registered id (not a direct UnbalancedDisk(...)
    instantiation) so the environment is wrapped in a TimeLimit (300 steps).
    Without a time limit the episode never ends, the env never resets, and
    stable-baselines3 has no episode boundaries to log -> no reward curve in
    TensorBoard.

    Monitor records per-episode reward and length, which is what populates
    'rollout/ep_rew_mean' and 'rollout/ep_len_mean' in TensorBoard.
    """
    env = gym.make("unbalanced-disk-v0", dt=0.025, umax=3.0)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Standalone evaluation (used for a final report after training)
# ---------------------------------------------------------------------------
def evaluate(model, env, num_episodes=10):
    """
    Run the (deterministic) policy for a number of episodes and return the
    mean total reward per episode.
    """
    total_rewards = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return float(np.mean(total_rewards))


def main():
    # Separate environments for training and for periodic evaluation, so eval
    # episodes do not interfere with the training rollouts.
    train_env = make_env()
    eval_env = make_env()

    # A2C with the default MLP policy. device="cpu" is correct here: the network
    # is tiny and the bottleneck is the scipy ODE solve inside each env step,
    # so a GPU would not help.
    model = sb3.A2C(
        "MlpPolicy",
        train_env,
        verbose=0,
        device="cpu",
        tensorboard_log=str(TB_LOG_DIR),
    )

    # EvalCallback periodically rolls out the current policy on eval_env and
    # logs the result to TensorBoard ('eval/mean_reward'). Whenever it sees a
    # new best mean reward it saves the model to best_model_save_path as
    # 'best_model.zip'. This is the correct way to pass evaluation to learn():
    # a *callback object*, not the return value of a function call.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_MODEL_DIR),
        log_path=str(EVAL_LOG_DIR),
        eval_freq=4000,          # run an evaluation every 10k training steps
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Run both callbacks together.
    callbacks = CallbackList([eval_callback])

    # Train. progress_bar=True needs the 'tqdm' and 'rich' packages
    # (set it to False if they are not installed).
    model.learn(
        total_timesteps=50000,
        callback=callbacks,
        tb_log_name="a2c",
        progress_bar=True,
    )

    # Save the final (last) model. The best model seen during training was
    # already saved by EvalCallback at BEST_MODEL_DIR / "best_model.zip".
    model.save(str(FINAL_MODEL_PATH))
    print(f"Final model saved to: {FINAL_MODEL_PATH}.zip")
    print(f"Best model saved to : {BEST_MODEL_DIR / 'best_model.zip'}")

    # Prefer the best checkpoint for evaluation/visualisation; fall back to the
    # final model if (e.g. for a very short run) no eval ever triggered.
    best_model_path = BEST_MODEL_DIR / "best_model.zip"
    if best_model_path.exists():
        model = sb3.A2C.load(str(best_model_path), device="cpu")
        print("Loaded best model for evaluation.")

    # Final evaluation report on a fresh eval env.
    mean_reward = evaluate(model, eval_env, num_episodes=10)
    print(f"\nFinal mean reward over 10 episodes: {mean_reward:.2f}")

    train_env.close()
    eval_env.close()

    # -----------------------------------------------------------------------
    # Visualise the trained policy
    # -----------------------------------------------------------------------
    # A human-rendered env to watch the swing-up. render_mode="human" opens a
    # pygame window.
    vis_env = gym.make("unbalanced-disk-v0", dt=0.025, umax=3.0, render_mode="human")
    model = sb3.A2C.load(str(best_model_path), device="cpu")
    obs, info = vis_env.reset()
    try:
        for _ in range(200):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vis_env.step(action)
            print(obs, reward)
            vis_env.render()
            time.sleep(1 / 24)
            if terminated or truncated:
                obs, info = vis_env.reset()   # unpack the (obs, info) tuple
    finally:  # always run, even on Ctrl-C, so the window/resources are released
        vis_env.close()


if __name__ == "__main__":
    main()
