# FUNCTIONS.md — DiTreeHist Project: Full API & Internals

This document covers the full public API and internal behavior of `ditreeHist_final.py`.  
It explains the purpose, inputs/outputs, tensor and array shapes, side-effects, common pitfalls, and performance notes for **every** function and class.

- Python ≥3.9, PyTorch ≥1.13 (tested with AMP), NumPy, Matplotlib.

---

## Table of Contents

1. [Config & Globals](#config--globals)
2. [Utilities](#utilities)
   - `set_seed`
   - `sinusoidal_time_embedding`
   - `crop_local_map`
   - `crop_local_map_torch`
   - Radius & Distance: `euclidean_distance`, `sample_goal_within_radius`
   - Normalization helpers: `normalize_map01_to_pm1`, `normalize_goal_xy_to_pm1`, `denorm_goal_pm1_to_xy`, `normalize_actions_to_pm1`, `denorm_actions_from_pm1`
   - Tensor shape helper: `ensure_nchw_single_channel`
3. [Dataset](#dataset)
   - `NpzMapDataset` (`__init__`, `_scan_files`, `__len__`, `__getitem__`)
4. [EMA](#ema)
   - `EMAModel` (`__init__`, `update`, `copy_to`, `state_dict`, `load_state_dict`)
5. [Model](#model)
   - `LocalMapEncoder` (`forward`)
   - `ActionHistoryEncoder` (`forward`)
   - `ConditionalDiffusionModel` (`forward`)
6. [Diffusion Core](#diffusion-core)
   - `DiffusionScheduler` (`__init__`, `q_sample`, `p_sample_step`)
7. [Training & Builders](#training--builders)
   - `build_dataloaders`
   - `build_model_and_sched`
   - `rollout_states_torch`
   - `train`
8. [Environment](#environment)
   - `CarState` (+ `to_array`)
   - `GridCarEnv` (`__init__`, `in_bounds`, `collision`, `step`, `sample_local_start_goal` [static-style])
9. [Planner (DiTree)](#planner-ditree)
   - `Node`
   - `DiffusionTreePlanner` (`__init__`, `_select_node`, `build_tree`, `_collect_history`, `_collect_action_history`, `sample_actions`, `sample_actions_k`, `_simulate_segment`)
10. [Inference / Evaluation / Visualization](#inference--evaluation--visualization)
    - `load_checkpoint`
    - `extract_best_path`
    - `_node_chain`
    - `extract_path_xy_from_node`
    - `topk_candidate_nodes`
    - `depth`
    - `collect_edges`
    - `visualize_map_and_paths`
    - `evaluate_over_testset`
    - `run_eval_once`
11. [Main / CLI](#main--cli)
12. [Data Shapes & Types Summary](#data-shapes--types-summary)
13. [Known Caveats & Tips](#known-caveats--tips)

---

## Config & Globals

The file defines a `DEFAULT_CONFIG` dictionary (module-global) with defaults for training and inference:

- **Data**: `train_data_dir`, `test_data_dir`, `map_crop_size`, `history_size`, `pred_horizon`, `action_dim`, `obs_dim`
- **Radius control**: `goal_radius`, `goal_radius_min`, `force_radius`
- **Training**: `batch_size`, `epochs`, `lr`, `num_workers`, `ema_decay`, `use_ema_for_eval`
- **Rollout loss**: `rollout_weight`, `rollout_yaw_weight`
- **Diffusion**: `diffusion_T`, `beta_start`, `beta_end`
- **Tree/env**: `dt`, `wheelbase`, `action_horizon`, `max_speed`, `max_steer`, `goal_tolerance`, `max_iterations`
- **Eval/viz**: (implicit) `viz_topk_paths`, `collision_substep_max_dist`
- **Checkpoints**: `ckpt_dir`, `model_name`, `history_mode`

*Side-effects*: none.  
*Pitfalls*: ensure `map_crop_size` matches your CNN assumptions; if you change it, retrain.

---

## Utilities

### `set_seed(seed: int) -> None`
**Purpose**: Deterministic-ish runs across Python/NumPy/Torch.

- **Inputs**: `seed` (`int`)
- **Outputs**: none
- **Side-effects**: sets `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- **Pitfalls**: Does not freeze CUDA kernels fully; numerical nondeterminism may remain.

---

### `sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor`
**Purpose**: Standard sinusoidal embedding for diffusion timestep conditioning.

- **Inputs**:
  - `t`: `(B,)` `LongTensor` of timesteps
  - `dim`: embedding dimension (int)
- **Outputs**: `(B, dim)` `FloatTensor`
- **Notes**: Pads if `dim` is odd.
- **Pitfalls**: `t.device` must match model device.

---

### `crop_local_map(grid: np.ndarray, cx: float, cy: float, crop: int) -> np.ndarray`
**Purpose**: Extract `crop × crop` patch around `(cx,cy)` from a **NumPy** grid, padding with ones (obstacles). Converts to `[-1,1]`.

- **Inputs**:
  - `grid`: `(H, W)` `uint8`/int with {0=free, 1=obstacle}
  - `cx, cy`: floats (world/grid coords; rounded to nearest cell)
  - `crop`: int (patch size)
- **Outputs**: `(1, crop, crop)` `float32` in `[-1,1]`
- **Pitfalls**: If near borders, patch is padded with `1` (obstacles) before normalization; that becomes `+1` after `[-1,1]`.

---

### `crop_local_map_torch(grid_t: torch.Tensor, cx: float, cy: float, crop: int) -> torch.Tensor`
**Purpose**: Torch version to avoid CPU↔GPU copies during inference, also returned in `[-1,1]`.

- **Inputs**:
  - `grid_t`: `(H, W)` `FloatTensor` on target device with values {0.,1.}
  - `cx, cy`: floats
  - `crop`: int
- **Output**: `(1, crop, crop)` `FloatTensor` in `[-1,1]`
- **Pitfalls**: Must pass a **single-channel** 2D tensor; not `(1,H,W)`.

---

### `euclidean_distance(a_yx: Tuple[int,int], b_yx: Tuple[int,int]) -> float`
**Purpose**: Euclidean distance in **grid cells** between `(y,x)` points.  
**Output**: scalar float.

---

### `sample_goal_within_radius(grid, start_yx, r_max, r_min=0.0, max_tries=500) -> Tuple[int,int]`
**Purpose**: Sample a free-cell goal within the Euclidean annulus `[r_min, r_max]` around `start`.

- **Inputs**: `grid (H,W)`, `start_yx (int,int)`, radii in cells
- **Output**: `(gy, gx)` (ints)
- **Side-effects**: None
- **Pitfalls**: If no free cells satisfy the annulus, raises `RuntimeError`. Make sure `r_max` is feasible for your maps.

---

### Normalization helpers

#### `normalize_map01_to_pm1(local: np.ndarray) -> np.ndarray`
- **In/Out**: `(1,H,W)` `{0,1}` → `[-1,1]` (float32).

#### `normalize_goal_xy_to_pm1(gx: float, gy: float, W: int, H: int) -> (float, float)`
- **Purpose**: Normalize 0-based pixel coordinates to `[-1,1]` over width/height.
- **Pitfalls**: Uses `(W-1)` and `(H-1)` denominators, consistent with image indices.

#### `denorm_goal_pm1_to_xy(gx_n, gy_n, W, H) -> (float, float)`
- Inverse mapping (not used in main flow).

#### `normalize_actions_to_pm1(actions: np.ndarray, max_speed: float, max_steer: float) -> np.ndarray`
- **In/Out**: `(...,2)` `[v, steer]` physical → `[-1,1]` (clipped)

#### `denorm_actions_from_pm1(actions_n: torch.Tensor, max_speed: float, max_steer: float) -> torch.Tensor`
- **In/Out**: `(B,T,2)` normalized → physical; clamps to limits.
- **Pitfalls**: Expects contiguous tensor; returns a new tensor.

---

### `ensure_nchw_single_channel(x: torch.Tensor) -> torch.Tensor`
**Purpose**: Guarantee `(B,1,H,W)` for map input.

- **Accepts**:
  - `(B,H,W)` → unsqueezes to `(B,1,H,W)`
  - `(B,H,W,1/3)` → permutes to `(B,1/3,H,W)`, then trims to 1 channel if needed
- **Pitfalls**: Will silently drop extra channels (`x = x[:, :1, :, :]`), which is intended here.

---

## Dataset

### `class NpzMapDataset(torch.utils.data.Dataset)`

**Purpose**: Windowed supervised samples from `.npz` logs.

**Expected NPZ keys**:
- `grid`: `(H,W)` `uint8` (0 free, 1 obstacle)
- `start_pos`: `(2,)` `(y,x)` (not used by dataset itself)
- `goal_pos`: `(2,)` `(y,x)`
- One or more trajectories under keys named like `optimal_path`, `suboptimal_path_*`, or ending with `path`. Each is `(T,6)` `[x,y,yaw,v,throttle,steer]`.

#### `__init__(root_dir, history_size, pred_horizon, map_crop_size, obs_dim=6)`
- **Indexes** `*.npz`.
- Stores sizes; builds index in `_scan_files`.

#### `_scan_files(self) -> None`
- For each log and each trajectory key, creates windows where `T - (history_size + pred_horizon) > 0`.
- **Side-effects**: populates `self._index` with `(file_idx, key, t0)`.

#### `__len__(self) -> int`
- Number of windows.

#### `__getitem__(self, i: int) -> Dict[str, torch.Tensor]`
Builds one training sample:

- **Loads**: `grid (H,W)`, `goal (y,x)`, `traj (T,6)`
- **Slices**:
  - `hist`: `(H,6)` last H obs
  - `future`: `(P,6)` next P obs
  - `actions`: `future[:, [3,5]]` → `(P,2)` `[v, steer]`
  - `action_history`: `hist[:, [3,5]]` → `(H,2)`
- **Local map**: `crop_local_map(...)` → `(1,crop,crop)` in `[-1,1]`
- **Normalize**:
  - `action_history_n`, `actions_n` → `[-1,1]` by limits
  - `goal (gx,gy)` to `[-1,1]` using image dims
- **Returns**:
  - `action_history`: `(H,2)` float32 (normalized)
  - `actions`: `(P,2)` float32 (normalized; diffusion target)
  - `local_map`: `(1,crop,crop)` float32 in `[-1,1]`
  - `goal`: `(2,)` float32 normalized
  - `last_state`: `(6,)` float32 **physical**
  - `future_states`: `(P,3)` `[x,y,yaw]` **physical** (for rollout loss)

*Pitfalls*: dataset normalizes actions & goal for the diffusion objective, but **not** states (rollout uses physical units). Keep this consistent in training.

---

## EMA

### `class EMAModel`
**Purpose**: Exponential moving average (EMA) of model parameters, used for stable evaluation.

#### `__init__(model, decay=0.999, device=None)`
- Creates `shadow` dict with cloned parameters.
- **Pitfalls**: Only parameters with `requires_grad=True` are tracked.

#### `update(model)`
- **No grad**: `ema = decay * ema + (1-decay) * param`.

#### `copy_to(model)`
- Copies shadow into `model` (in-place).
- **Side-effects**: Overwrites model parameters.

#### `state_dict()` / `load_state_dict(state)`
- Save/load EMA shadow (and decay).

*Performance note*: EMA update cost is proportional to parameter count; negligible vs backprop.

---

## Model

### `class LocalMapEncoder(nn.Module)`
**Purpose**: CNN encoder for local map patch.

- **Input**: `(B,1,crop,crop)` float in `[-1,1]`
- **Output**: `(B,128)` (default `out_dim=128`)
- **Layers**: Conv-ReLU → Conv-ReLU → AvgPool → Conv-ReLU → AvgPool → Conv-ReLU → GAP → FC-ReLU
- **Pitfalls**: Single-channel only.

#### `forward(x)`
- x: `(B,1,H,W)` → `(B,128)`

---

### `class ActionHistoryEncoder(nn.Module)`
**Purpose**: GRU over action history.

- **Input**: `(B,H,2)` `[v, steer]` (normalized)
- **Output**: `(B,hidden_size)` (default 128)

#### `forward(action_seq)`
- Returns last hidden state.

---

### `class ConditionalDiffusionModel(nn.Module)`
**Purpose**: Predict diffusion noise `ε` for an **action sequence** conditioned on `(map, action history, goal, time)`.

- **Init args**: `obs_dim`, `history_size`, `pred_horizon`, `map_crop`, `cond_dim=128`, `time_dim=64`, `hidden=256`, `action_dim=2`
- **Submodules**:
  - `map_enc: LocalMapEncoder`
  - `hist_enc: ActionHistoryEncoder`
  - `goal_fc: Linear(2→cond_dim)+ReLU`
  - `t_fc: Linear(time_dim→cond_dim)+ReLU`
  - `fuse: MLP(4*cond_dim → hidden → hidden)`
  - `action_in: Linear(P*action_dim → hidden)`
  - `out: Linear(hidden → P*action_dim)`

#### `forward(noisy_actions, history, goal, local_map, t)`
- **Inputs**:
  - `noisy_actions`: `(B, P*2)` float
  - `history`: `(B,H,2)` normalized
  - `goal`: `(B,2)` normalized
  - `local_map`: `(B,1,crop,crop)` in `[-1,1]`
  - `t`: `(B,)` long
- **Output**: `eps_hat`: `(B, P*2)`
- **Computation**: encodes all conditions, fuses, adds to `action_in(noisy_actions)`, MLP → noise.

*Pitfalls*: All tensors must share the same device/dtype; shapes must match `pred_horizon`.

---

## Diffusion Core

### `class DiffusionScheduler`
**Purpose**: Linear beta schedule; forward and one reverse denoise step.

#### `__init__(T, beta_start, beta_end, device)`
- Precomputes `betas`, `alphas`, `alphas_cumprod`, and their square roots.

#### `q_sample(x0, t, noise=None) -> (x_t, noise)`
- **Inputs**:
  - `x0`: `(B,D)` clean target (here `D=P*2`)
  - `t`: `(B,)` long
  - `noise` optional `(B,D)`
- **Output**: `x_t = sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) ε`, and `noise`

#### `p_sample_step(model, x_t, t, cond_kwargs) -> x_{t-1}`
- **One step** reverse diffusion using predicted `ε`:
  - `mean = 1/√α_t * ( x_t - (β_t / √(1-ᾱ_t)) * ε̂ )`
  - Adds `σ_t z` if `t>0`
- **Inputs**:
  - `model`: callable `f(x_t, **cond_kwargs, t=t)`
  - `x_t`: `(B,D)`
  - `t`: `(B,)`
  - `cond_kwargs`: dict passed to model
- **Output**: `(B,D)` `x_{t-1}`

*Pitfalls*: `t` must be a tensor (not Python int) on device.

---

## Training & Builders

### `build_dataloaders(cfg) -> DataLoader`
- Builds `NpzMapDataset` from `cfg["train_data_dir"]`.
- **Respects**: `batch_size`, `num_workers`, `drop_last=True`.

---

### `build_model_and_sched(cfg) -> (ConditionalDiffusionModel, DiffusionScheduler)`
- Creates model on `cfg["device"]` and scheduler with linear beta schedule.

---

### `rollout_states_torch(x0, actions, dt=0.2, wheelbase=1.0, max_steer=None, max_speed=None) -> torch.Tensor`
**Purpose**: Differentiable bicycle model rollout for loss.

- **Inputs**:
  - `x0`: `(B,6)` (uses `[x,y,yaw]`)
  - `actions`: `(B,T,2)` physical `[v, steer]`
- **Output**: `(B,T,3)` predicted `[x,y,yaw]`
- **Notes**: Time loop in PyTorch; clamps `v`/`steer` to limits if provided.
- **Pitfalls**: Avoid in-place ops on autograd graph; function returns a new tensor.

---

### `train(cfg) -> None`
**Purpose**: Full training loop with EMA and AMP.

- **Flow**:
  1. Seeds & dirs.
  2. Build model/scheduler/optimizer, **EMA** (`ema_decay`) and LR scheduler (cosine + warmup).
  3. Enable AMP if CUDA (`torch.amp.autocast('cuda')` + `torch.amp.GradScaler('cuda')`).
  4. For each epoch/batch:
     - Load batch tensors to device.
     - `x0_actions = actions_norm.view(B, P*2)`
     - Sample `t`; forward `q_sample` → `(x_t, noise)`
     - Model predicts `eps_hat`
     - **ε-loss**: `MSE(eps_hat, noise)`
     - Reconstruct `x0_hat` and **denormalize** to physical actions
     - Rollout → `(B,P,3)`; compute **rollout loss** on `[x,y]` + yaw (weighted)
     - Total loss: `loss = loss_eps + λ * loss_roll`
     - Backprop w/ AMP, grad clip, optimizer step
     - **EMA.update(model)**
  5. Save checkpoint with `{"model", "cfg", "ema"}`

- **Inputs**: via `cfg`, dataset on disk
- **Outputs**: writes checkpoint files
- **Side-effects**: prints stats, writes CKPT
- **Pitfalls**:
  - Ensure dataset actions/goals are normalized; states are physical.
  - Keep `pred_horizon` consistent across dataset/model.
  - Use `ensure_nchw_single_channel` for `local_map`.
  - AMP usage must follow current PyTorch API (`torch.amp`).

*Performance notes*:  
- EMA slightly improves validation and stability.  
- Rollout loss improves physical coherence but increases compute.

---

## Environment

### `@dataclass CarState`
- Fields: `x, y, yaw, v, throttle, steer` (floats)
- `to_array()` → `(6,)` `float32`

---

### `class GridCarEnv`
Grid world + bicycle dynamics + collision.

#### `__init__(grid, dt=0.2, wheelbase=1.0, max_speed=2.0, max_steer=0.6)`
- Stores `grid` `(H,W)` `uint8`, sizes, `dt`, `L`, limits.

#### `in_bounds(x, y) -> bool`
- Rounds `(x,y)` and checks inside bounds.

#### `collision(x, y) -> bool`
- Returns `True` if out-of-bounds **or** cell is obstacle `1`.

#### `step(state: CarState, action: Tuple[float,float]) -> (CarState, bool)`
- Bicycle update with clamped commands.
- Returns `(next_state, collided_flag)` based on **post-step** position.

#### `sample_local_start_goal(grid, max_dist=5, try_start_from_given=None, max_tries=200) -> ((sy,sx),(gy,gx))`
- Samples free start/goal within **Manhattan** distance ≤ `max_dist`.
- **Note**: static-style method (no `self`).

*Pitfalls*: The planner’s short-range sampling uses **Euclidean** radius helpers (above), while this helper is **Manhattan**; do not confuse them.

---

## Planner (DiTree)

### `class Node`
- Represents a node in the search tree.
- Fields:
  - `state: CarState`
  - `parent: Optional[Node]`
  - `path_from_parent: Optional[np.ndarray]` `(K,6)` — states along executed edge
  - `actions_from_parent: Optional[np.ndarray]` `(K-1,2)` — actions used to get here
  - `children: List[Node]`

---

### `class DiffusionTreePlanner`
Holds env, model, scheduler, config; performs tree expansion guided by diffusion.

#### `__init__(env, model, sched, cfg)`
- Stores references, device, and `grid_torch` `(H,W)` float tensor on device for torch-crops.

#### `_select_node(self, nodes: List[Node]) -> Node`
- Returns a node to expand.
- Current policy: uniform random choice (stochastic exploration).
- *Performance note*: You can bias toward frontier/nearest-to-goal nodes to speed planning.

#### `build_tree(self, start: (int,int), goal: (int,int), max_iterations=None) -> (List[Node], Optional[Node])`
- **Purpose**: Construct diffusion-guided tree.
- **Process**:
  - Build root with yaw toward goal.
  - For up to `max_iterations`:
    - Pick node via `_select_node`.
    - Collect action history (H).
    - `actions = sample_actions_k(...)` (K candidates; pick best by simulated progress).
    - Simulate segment via `_simulate_segment` (with sub-step collision checking).
    - If valid, append child node and check goal tolerances.
    - **Stop early** if goal reached (returns a `goal_node`).
- **Outputs**: list of all nodes and the goal node (if found).

*Pitfalls*: Ensure `goal_tolerance` is in **grid cells** (same units as planner coordinates).

#### `_collect_history(self, node, limit: int) -> List[np.ndarray]`
- Legacy: collect up to `limit` past **states** from root to node (stitching edges).
- **Output**: list of `(6,)` arrays.

#### `_collect_action_history(self, node, H: int) -> np.ndarray`
- Gather last `H` actions `[v, steer]` along path root→node, padding with zeros if insufficient depth.
- **Output**: `(H,2)` `float32` **physical** (model normalizes inside `sample_actions`).

#### `sample_actions(self, node, goal: (float,float), act_hist: Optional[np.ndarray]) -> np.ndarray`
- **Purpose**: Generate a single action sequence via diffusion.
- **Steps**:
  1. Local map crop around current `(x,y)` → `(1,1,crop,crop)` in `[-1,1]`.
  2. Action history (H) → normalize to `[-1,1]`, apply `history_mode` (`full/last/none`).
  3. Normalize goal `(gx,gy)` to `[-1,1]`.
  4. Reverse diffusion loop (`T` steps) to produce normalized actions `(P,2)`.
  5. **Denormalize** to physical limits; **truncate** to `A = action_horizon`; clamp to limits.
- **Output**: `(A,2)` NumPy array (physical).

*Pitfalls*:
- All conditioning must be on the same device.
- `pred_horizon ≥ action_horizon`.
- History mode `none` can be a strong baseline; compare metrics with `full`.

#### `sample_actions_k(self, node, goal, act_hist=None, K=4) -> np.ndarray`
- **Purpose**: Try `K` stochastic diffusion samples and pick the best **scored** candidate.
- **Scoring**:
  - Uses `_simulate_segment(node.state, actions)` for **consistent** collision checking (with sub-steps).
  - If valid, compute `d0 = dist(start, goal)` and `d1 = dist(end, goal)`, score = `(d0 - d1) - 0.01*len(actions)`.
  - Pick max score; fallback to `sample_actions` if all invalid.
- **Output**: `(A,2)` physical actions.

*Pitfalls*: Keep `K` small for speed; larger K improves local optimality but costs more.

#### `_simulate_segment(self, start: CarState, actions: np.ndarray) -> (np.ndarray, bool)`
- **Purpose**: Robust segment simulation with **sub-sampling** along each motion to avoid skipping thin obstacles.
- **Inputs**:
  - `start`: `CarState`
  - `actions`: `(A,2)` physical
- **Behavior**:
  - For each action:
    - First compute the next state with `env.step` (single integrator step).
    - Then sub-sample the **straight segment** between `(x,y)` and `(x',y')` with spacing `collision_substep_max_dist` (cfg; default `0.5` cells).
    - If any sub-sample hits an obstacle or OOB → **invalid**.
  - Accumulate states; return `(K,6)` trajectory (including start), and `valid: bool`.
- **Pitfalls**:
  - Sub-step spacing too large can miss thin barriers; too small slows down.
  - Sub-sampling uses straight interpolation in `(x,y)` (consistent with our simple kinematics step).

---

## Inference / Evaluation / Visualization

### `load_checkpoint(ckpt_path: str, cfg: Dict) -> (ConditionalDiffusionModel, DiffusionScheduler)`
- Loads model and scheduler, merges saved `cfg` with current `cfg`.
- If `use_ema_for_eval=True` and `"ema"` exists in ckpt, loads EMA shadow and copies to model for evaluation.
- Returns `(model.eval(), sched)`.

*Pitfalls*: Merging configs: current `cfg` values override saved ones by design.

---

### `extract_best_path(goal_node: Optional[Node], nodes: List[Node], goal_xy: (float,float)) -> List[(float,float)]`
- If `goal_node` exists: stitch chain root→goal.  
- Else: pick node with smallest Euclidean distance to `goal_xy` and stitch chain root→that node.  
- **Output**: list of `(x,y)` points.

---

### `_node_chain(n: Node) -> List[Node]`
- Utility: returns list of nodes from root to `n`.

### `extract_path_xy_from_node(n: Node) -> List[(float,float)]`
- Stitches `(x,y)` along chain root→`n` without duplicating joints.

---

### `topk_candidate_nodes(nodes: List[Node], goal_xy: (float,float), k: int=5) -> List[Node]`
- Prefers **leaves**; scores by Euclidean distance to goal, tie-breaking by deeper depth.
- Returns up to `k` nodes (unique, may exclude the goal node if you already include it).

### `depth(n: Node) -> int`
- Node depth from root.

---

### `collect_edges(nodes: List[Node]) -> List[List[(float,float)]]`
- Extracts every edge polyline (list of `(x,y)` states along `path_from_parent`).

---

### `visualize_map_and_paths(grid, edges, paths, out_png, start_xy=None, goal_xy=None, path_labels=None) -> None`
- Saves a PNG:
  - Map with **free=light-gray**, **obstacle=black**.
  - All tree edges (light blue).
  - Multiple candidate paths (distinct colors).
  - Start/Goal markers.
  - Legend **outside** the map.
- **Inputs**:
  - `grid`: `(H,W)` `uint8`
  - `edges`: `List[List[(x,y)]]`
  - `paths`: `List[List[(x,y)]]` (e.g., best + top-k candidates)
  - `path_labels`: optional list of labels
- **Side-effects**: writes `out_png`.

*Pitfalls*: Axes are image coordinates; Y is inverted (handled by `imshow` defaults).

---

### `evaluate_over_testset(cfg: Dict, out_dir: str, visualize: bool=True) -> str`
- Iterates over all test NPZs:
  1. Load grid and dataset `start/goal` (unless randomized/forced radius).
  2. Optional radius logic:
     - `random_pair`: random free start, goal sampled within radius.
     - `random_local_goal`: keep dataset start, sample goal within radius.
     - `force_radius`: if dataset (start, goal) violates annulus, resample goal.
     - Logs Euclidean start-goal distance.
  3. Build env, load checkpoint (`EMA` for eval if available), create planner.
  4. Build tree (early stop on goal reach).
  5. Extract edges and **multiple** candidate paths (goal node + top-k leaves).
  6. Save JSON/CSVs/PNGs per map and a summary CSV.
- **Returns**: path to `summary.csv` (or `""` if failure).
- **Side-effects**: writes outputs.

*Pitfalls*: `viz_topk_paths` default is 5; reduce if output cluttered. Loading ckpt repeatedly across files is fine but you can move it outside the loop for speed.

---

### `run_eval_once(cfg: Dict, out_dir: str, visualize: bool=True) -> str`
- Thin wrapper around `evaluate_over_testset`.

---

## Main / CLI

### `main()`
CLI flags (parsed by `argparse`):

- `--epochs INT` (default 0): train epochs (0 = skip training)
- `--eval_all_tests` (flag): run evaluation on **all** test NPZs
- `--visualize` (flag): save visualizations
- `--out_dir STR` (default `outputs_test_history`): evaluation outputs
- `--pipeline` (flag): do “optional train → eval none → eval full”
- `--out_dir_none STR` (default `meta_outputs_test_no_history`): pipeline dir (history off)
- `--out_dir_full STR` (default `meta_outputs_test_history`): pipeline dir (history on)
- `--history_mode {full,last,none}` (default `full`): inference conditioning
- `--random_local_goal` (flag): keep dataset start, re-sample goal within radius
- `--random_pair` (flag): re-sample start and goal within radius
- `--max_local_dist INT` (default 5): **Manhattan** distance for legacy helper (used only if those random flags are active in the old path; the radius-based logic is the recommended one)

Behavior:
- Pipeline mode: optional train, then two eval passes (none/full).
- Else: train if `epochs>0`, then either eval-all or single-demo.

---

## Data Shapes & Types Summary

- **Grid**: `(H,W)` `np.uint8` with `{0,1}`
- **Trajectory row**: `(6,)` `[x,y,yaw,v,throttle,steer]` `float32`
- **History window (states)**: `(H,6)` *(not used by model)*
- **Action history**: `(H,2)` `[v, steer]` **normalized** in dataset (for model); **physical** inside planner’s collector
- **Future actions (target)**: `(P,2)` **normalized**
- **Future states (GT)**: `(P,3)` `[x,y,yaw]` **physical**
- **Local map**: `(1,crop,crop)` float in `[-1,1]`
- **Goal**: `(2,)` normalized `[-1,1]`
- **Model noisy input / output**: `(B, P*2)`
- **Planner sampled actions**: `(A,2)` physical (`A = action_horizon`)
- **Simulated edge states**: `(K,6)` (includes start)

---

## Known Caveats & Tips

- Be consistent: dataset **normalizes** actions/goals; planner **denormalizes** predicted actions for rollout and execution.
- `sample_actions_k` now uses `_simulate_segment` for scoring to keep consistency with collision checks.
- EMA improves eval stability; enable `use_ema_for_eval`.
- If success rates are low: use small Euclidean radii; ensure `map_crop_size` is large enough; check `history_mode` and try `none` as baseline; consider training with more data or longer epochs.
- AMP: The code uses `torch.amp` API (`autocast('cuda')`, `GradScaler('cuda')`).

