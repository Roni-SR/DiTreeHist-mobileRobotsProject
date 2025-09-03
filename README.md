# DiTree – Diffusion-Based Tree Expansion for Motion Planning

This repository implements **DiTree**, a diffusion-conditioned tree expansion algorithm for motion planning in grid maps.  
It combines **conditional diffusion models** for action sequence prediction with **tree search** for robust path planning under obstacles.

---

## Features
- **Conditional Diffusion Model**:
  - Inputs: local cropped map, action history, goal, timestep embedding
  - Outputs: predicted action sequence
- **Action History Encoder (GRU)** with multiple history modes:
  - `full` – full action history  
  - `last` – last action repeated  
  - `none` – zero vector (baseline)
- **Local Map Encoder** (CNN) for occupancy crops
- **Diffusion Scheduler** with configurable β schedule
- **EMA (Exponential Moving Average)** for stable training & better inference
- **Bicycle Kinematic Rollout** for differentiable trajectory simulation
- **Collision Checking with Sub-sampling**
- **Evaluation & Visualization**:
  - Saves JSON/CSV/PNG outputs
  - Best path extraction
  - Tree expansion visualization

---

## Repository Structure
.
├── ditreeHist_final.py # Main pipeline: training, evaluation, inference
├── compareRuns.py # Compare performance with/without history
├── npztest.py # Quick dataset inspection & test
├── aStarPID_gen.py # Data generator (A* + PID baseline)
├── checkpoints/ # Saved model weights
├── results/ # Evaluation results, CSVs, visualizations
├── results_and_codes/ # Bundled results & codes archive
└── README.md # Project documentation

---

## Installation
Clone and install dependencies:
```bash
git clone https://github.com/Roni-SR/DiTreeHist-mobileRobotsProject.git
cd ditree
pip install -r requirements.txt


---

## Requirements
Python ≥ 3.8  ;  PyTorch ≥ 2.0  ;  NumPy, Matplotlib, tqdm

---

## Training
To train the model on the provided dataset:
python ditreeHist_final.py --epochs 100 --eval_all_tests --visualize

Key Training Parameters:
--epochs : number of training epochs
--batch_size : training batch size
--lr : learning rate
--ema_decay : EMA decay factor (default: 0.999)
--rollout_weight : weight for rollout loss
--rollout_yaw_weight : additional weight for yaw loss

---

## Inference & Evaluation
Single Demo on First Test Map
python ditreeHist_final.py --visualize

Evaluate All Test Maps
python ditreeHist_final.py --eval_all_tests --visualize --out_dir results_eval

Full Pipeline (Train + Eval None + Eval Full)
python ditreeHist_final.py --epochs 100 --pipeline --visualize


This produces results in:
-meta_outputs_test_no_history/
-meta_outputs_test_history/

---

## Important Config Parameters
Defined in DEFAULT_CONFIG (inside ditreeHist_final.py):
| Parameter               | Description                                   |
| ----------------------- | --------------------------------------------- |
| `map_crop_size`         | Size of cropped local map (pixels)            |
| `history_size`          | Length of action history (timesteps)          |
| `pred_horizon`          | Number of future actions to predict           |
| `action_horizon`        | Number of actions executed per tree expansion |
| `goal_tolerance`        | Euclidean tolerance to goal (grid units)      |
| `max_iterations`        | Maximum iterations for tree expansion         |
| `diffusion_T`           | Number of diffusion steps                     |
| `beta_start`/`beta_end` | Diffusion β schedule                          |
| `ema_decay`             | EMA decay (stability of training)             |
| `use_ema_for_eval`      | Whether to use EMA weights at inference       |

---

## References
@inproceedings{rimon2024mamba,
  title     = {MAMBA: An Effective World Model Approach for Meta-Reinforcement Learning},
  author    = {Zohar Rimon and Tom Jurgenson and Orr Krupnik and Gilad Adler and Aviv Tamar},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2024},
  institution = {Technion -- Israel Institute of Technology, Ford Research Center Israel},
  url       = {https://github.com/zoharri/mamba}
}

@article{hassidof2023ditree,
  title     = {Train Once, Plan Anywhere: Kinodynamic Motion Planning via Diffusion Trees},
  author    = {Yaniv Hassidof and Tom Jurgenson and Kiril Solovey},
  journal   = {arXiv preprint arXiv:2307.XXXXX},
  year      = {2023},
  institution = {Technion -- Israel Institute of Technology},
  url       = {https://arxiv.org/abs/2307.XXXXX}
}

## Author
Developed as part of a project in Mobile Robots, 0460213, Electrical and Computer Engineering, Technion, Israel.
Maintained by: Roni Shakarov Reisser