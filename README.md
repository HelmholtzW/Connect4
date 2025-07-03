# Connect 4 – Deep Q-Learning Experiments

A compact research playground for experimenting with deep Q-learning variants on the classic **Connect 4** game.

*This project was created during my time at the University of Groningen as part of the Honours Program.*

The repository contains:

* A lightweight Connect 4 environment implemented in pure NumPy
* Several neural network-based agents (DQN, Q-V, Q-V-Advantage)
* Training scripts for local execution or SLURM clusters
* Example results & helper scripts for plotting learning curves

---

## 1. Project layout

```
Connect4/
├── code/                # All source files
│   ├── connect4.py      # Game environment (6×7 board)
│   ├── dqn_agent.py     # Classical Deep-Q-Network (DQN)
│   ├── dqvn_agent.py    # Separate Q & V networks (QV-Learning)
│   ├── dqva_agent.py    # Q, V & Advantage networks (QVA-Learning)
│   ├── training.py      # Generic training loop (QV by default)
│   ├── training_peregrine.py     # DQN training @ 50 k games
│   ├── training_qv_peregrine.py  # QV training @ 50 k games
│   ├── training_qva_peregrine.py # QVA training @ 50 k games
│   ├── plotting.py      # Utilities to create comparison plots
│   ├── *.sh             # SLURM batch helpers
│   └── requirements.txt # Exact python package versions
├── results/             # Example JSON scores & PNG plots
└── README.md            # (you are here)
```

---

## 2. Quick start

1. **Install dependencies**:

```bash
pip install -r code/requirements.txt
```

2. **Train locally** (example: DQN for 10 k episodes):

```bash
python code/training.py       # default QV agent
# or
python code/training_peregrine.py       # DQN
python code/training_qv_peregrine.py    # QV
python code/training_qva_peregrine.py   # QVA
```

Each script writes progress to `stats_*` (JSON) and prints intermediate win rates.

3. **Plot learning curves** (optional):

```bash
python code/plotting.py       # Generates comparison PNGs
```

---

## 3. Running on a SLURM cluster

Batch files in `code/*.sh` submit the training jobs with sensible defaults:

```bash
sbatch code/connect4_batch.sh   # DQN
sbatch code/qv_batch.sh         # QV
sbatch code/qva_batch.sh        # QVA
```

Adjust `--time`, memory, or the referenced python script as needed.

---

## 4. How the agents work (high level)

| Agent | Model architecture | Target update | Notes |
|-------|--------------------|--------------|-------|
| **DQN** (`dqn_agent.py`) | Flatten → Dense(50) sigmoid → Dense(7) linear | Soft-update every N games | Standard DDQN style |
| **QV** (`dqvn_agent.py`)  | Separate networks: Q(s,a) & V(s)             | V-network soft-update     | Learns state-value to stabilise Q |
| **QVA** (`dqva_agent.py`) | Q, V and Advantage networks                  | V soft-update             | Adds explicit Advantage estimator |

All agents share:

* Replay buffer with configurable size
* ε-greedy exploration (decays from 0.2 → 0.001)
* Discount factor γ = 1.0 (game is short & episodic)
* Soft-target coefficient 0.001

The training loop (`training*.py`) alternates between the learning agent and a handcrafted **bench player** that plays:

1. **Winning move** if one exists
2. **Blocking move** if the agent has a winning move next
3. A random valid move otherwise

---

## 5. Results directory

`results/` keeps historical experiments exactly as produced by prior runs (JSON scores + PNG plots). They are **not** required to run new experiments but serve as references.

---

## 6. Contributing / extensions

Feel free to experiment with:

* Different network sizes or activations (`create_*_model` functions)
* Alternative exploration schedules
* Self-play instead of the heuristic bench opponent

If you add a new training script or agent, remember to:

1. Use the same folder structure under `code/`.
2. Update this README with a short description (consult the project owner before large design changes [[user rule 6]]).

---

## 7. License

This project is released under the MIT License. See `LICENSE` (to be added) for details. 