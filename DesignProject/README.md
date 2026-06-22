# Design Project — Unbalanced Disk

System identification (Part 1) and control (Part 2) of the unbalanced-disk
benchmark. The simulation/hardware environment lives in `../gym-unbalanced-disk`.

```
DesignProject/
├── Part1_System_Identification/   # report Parts I–III
│   ├── P1_ANN.ipynb               # MLP & recurrent (RNN/LSTM) training + evaluation
│   ├── P1_GP.ipynb                # Gaussian-process training + evaluation
│   ├── models/                    # best trained models (GP, MLP, RNN, LSTM)
│   └── submission_files/          # prediction files submitted for the benchmark
│
└── Part2_Control/
    ├── Q_Learning/q_learning/
    │   ├── q-learning.ipynb       # tabular Q-learning training
    │   └── q_learning_sim.ipynb   # Q-learning simulation / control implementationks
    └── A2C_PPO/
        ├── P2_A2C_PPO.ipynb       # A2C & PPO training notebook (see its own README)
        └── models/
            ├── grid_search/       # models for report chapter IV — Advanced RL
            └── reference_tracking/# models for report chapter V — Reference tracking
```

## Part 1 — System Identification
- **`P1_ANN.ipynb`** walks through training and evaluating the artificial neural
  network models: a feed-forward **MLP** and the **recurrent** (RNN/LSTM)
  architectures.
- **`P1_GP.ipynb`** walks through training and evaluating the **Gaussian-process**
  model.
- Best-performing models are saved in `models/`; the files submitted to the
  benchmark are in `submission_files/`.

## Part 2 — Control
- **`Q_Learning/`** — `q-learning.ipynb` covers tabular Q-learning training;
  `q_learning_sim.ipynb` shows the simulation / control implementation.
- **`A2C_PPO/`** — `P2_A2C_PPO.ipynb` trains and evaluates A2C and PPO agents
  (see `A2C_PPO/README.md` for details). Trained models are split by report
  chapter: `models/grid_search/` (chapter IV, Advanced Reinforcement Learning)
  and `models/reference_tracking/` (chapter V, Reference Tracking).

## Getting started
**Python 3.12** is required (the pinned `requirements.txt` was built and tested on
3.12.13; the pinned `numpy`/`torch`/`pandas` versions need 3.12).

From the repository root, create a virtual environment and install everything in
one shot — `requirements.txt` is a full pin of a known-good environment and
includes the editable `gym-unbalanced-disk` package:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
This installs all packages used across the project:
- **Part 1 (ANN/GP):** `torch`, `scikit-learn`, `optuna`, `pandas`, `numpy`, `matplotlib`, `joblib`
- **Part 2 (Q-learning, A2C/PPO):** `stable-baselines3`, `gymnasium`, `torch`, `tensorboard`, `joblib`, `filelock`, `pygame`
- **Env package:** `gym-unbalanced-disk` (installed editable) — importing
  `gym_unbalanced_disk` registers the `unbalanced-disk-v0` /
  `unbalanced-disk-sincos-v0` gym env ids used throughout.

Then launch Jupyter and run the notebooks from within their folders.