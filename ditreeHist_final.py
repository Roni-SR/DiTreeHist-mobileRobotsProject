# -*- coding: utf-8 -*-
"""
DiTree (PyTorch-only) – diffusion-conditioned tree expansion over grid maps.

Enhancements in this version:
- Add --eval_all_tests: run inference on EVERY NPZ under test_data_dir and save per-map outputs.
- Visualization utilities: save PNG of the grid with the DiTree edges and the best path (no arrows).
- Save best path CSV per test map, and a summary CSV across the test set.

Expected NPZ keys (as produced by aStarPID_genLog.py):
- grid: (H, W) int, 0=free, 1=obstacle
- start_pos: (y, x) int
- goal_pos: (y, x) int
- optimal_path:  (T, 6) float -> [x, y, yaw, v, throttle, steer]
- suboptimal_path_k: same structure for k=1..K (optional)
"""
from __future__ import annotations

import os
import math
import glob
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ----------------------------- Config -----------------------------

DEFAULT_CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # data
    "train_data_dir": "NewPID_maps/train_maps",
    "test_data_dir":  "NewPID_maps/test_maps",
    "map_crop_size": 32,
    "history_size": 10,         # how many past states to condition on
    "pred_horizon": 12,         # model predicts this many actions per training sample
    "action_dim": 2,            # [v, steer]
    "obs_dim": 6,               # [x, y, yaw, v, throttle, steer]
    "goal_radius": 5.0,        # max radius to sample goal from start during data gen
    "goal_radius_min": 0.0,    # min radius to sample goal from start during data gen
    "force_radius": True,    # if True, resample start/goal until within radius
    "collision_substep_max_dist": 0.5,  # max dist between collision checks during data gen


    # training
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "num_workers": 2,
    "ema_decay": 0.999,         # EMA decay factor (typical: 0.999–0.9999)
    "use_ema_for_eval": True,   # Use EMA weights for inference if present

    # rollout loss
    "rollout_weight": 0.5,     # λ: weight for rollout loss (0 disables it)
    "rollout_yaw_weight": 0.1, # relative weight for yaw vs. position in rollout loss

    # diffusion
    "diffusion_T": 64,          # number of diffusion steps
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # tree / env
    "dt": 0.2,                  # simulation step time (s) 0.1
    "wheelbase": 1.0,           # bicycle model L
    "action_horizon": 8,        # steps to execute when expanding a node (<= pred_horizon) 6
    "max_speed": 2.0,
    "max_steer": 0.6,           # ~34 degrees
    "goal_tolerance": 1.5,      # distance to goal in grid units to stop
    "max_iterations": 1000,      # tree expansion iterations

    # checkpoints
    "ckpt_dir": "checkpoints_newLossMetaRNNNormImp2",
    "model_name": "ditree_pytorch_model.pth",
    "history_mode": "full",  # conditioning at inference: full/last/none

}

# ----------------------------- Utils -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(10000.0),
            steps=half,
            device=t.device
        ) * (-1.0 * 2.0 / half)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def crop_local_map(grid: np.ndarray, cx: float, cy: float, crop: int) -> np.ndarray:
    cx_i, cy_i = int(round(cx)), int(round(cy))
    h, w = grid.shape
    pad = crop // 2
    padded = np.ones((h + 2*pad, w + 2*pad), dtype=grid.dtype)
    padded[pad:pad+h, pad:pad+w] = grid
    y0 = cy_i
    x0 = cx_i
    cut = padded[y0:y0+crop, x0:x0+crop]
    if cut.shape != (crop, crop):
        cut = np.pad(
            cut,
            ((0, max(0, crop - cut.shape[0])), (0, max(0, crop - cut.shape[1]))),
            constant_values=1
        )
        cut = cut[:crop, :crop]
    # ALWAYS: (1,C,C) float32 and normalize 0/1 -> [-1,1]
    cut = cut.astype(np.float32)[None, ...]
    cut = normalize_map01_to_pm1(cut)
    return cut


def crop_local_map_torch(grid_t: torch.Tensor, cx: float, cy: float, crop: int) -> torch.Tensor:
    """
    Torch-based crop identical in logic to crop_local_map (NumPy):
    - pads with 1 (obstacles)
    - uses int(round(...)) for indices
    - returns shape (1, crop, crop) float32
    grid_t must be (H, W) float32 on the target device with values {0.,1.}
    """
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    H, W = grid_t.shape
    pad = crop // 2
    # F.pad: (left, right, top, bottom)
    padded = F.pad(grid_t.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="constant", value=1.0)
    y0, x0 = cy_i, cx_i
    cut = padded[0, 0, y0:y0+crop, x0:x0+crop]  # guaranteed crop size due to padding
    cut = cut.unsqueeze(0)  # (1, crop, crop)
    cut = (cut * 2.0) - 1.0  # 0/1 -> -1/+1
    return cut

# ---------- Radius-based utilities ----------

def euclidean_distance(a_yx: Tuple[int, int], b_yx: Tuple[int, int]) -> float:
    """Euclidean distance in grid cells between two (y,x) points."""
    dy = float(b_yx[0] - a_yx[0])
    dx = float(b_yx[1] - a_yx[1])
    return math.hypot(dx, dy)

def sample_goal_within_radius(grid: np.ndarray,
                              start_yx: Tuple[int, int],
                              r_max: float,
                              r_min: float = 0.0,
                              max_tries: int = 500) -> Tuple[int, int]:
    """
    Sample a free-cell goal within a Euclidean annulus around 'start_yx':
      r_min <= ||goal - start||_2 <= r_max  (in grid cells).
    Returns (gy, gx). Raises if failed.
    """
    H, W = grid.shape
    sy, sx = start_yx
    rng = np.random.default_rng()

    r_min = max(0.0, float(r_min))
    r_max = max(float(r_max), r_min + 1e-6)

    for _ in range(max_tries):
        # Area-uniform sampling in a disk (then shifted to [r_min, r_max])
        u = rng.random()
        r = math.sqrt((r_max**2 - r_min**2) * u + r_min**2)
        theta = rng.random() * 2.0 * math.pi
        gy = int(round(sy + r * math.sin(theta)))
        gx = int(round(sx + r * math.cos(theta)))
        if 0 <= gy < H and 0 <= gx < W and grid[gy, gx] == 0:
            return (gy, gx)

    # Fallback: deterministic scan in the bounding square
    r_ceiling = int(math.ceil(r_max))
    for dy in range(-r_ceiling, r_ceiling + 1):
        for dx in range(-r_ceiling, r_ceiling + 1):
            gy, gx = sy + dy, sx + dx
            if 0 <= gy < H and 0 <= gx < W and grid[gy, gx] == 0:
                d = math.hypot(dx, dy)
                if r_min <= d <= r_max:
                    return (gy, gx)

    raise RuntimeError("Failed to sample a GOAL inside the required radius")


# ---- Normalization helpers (limits mode) ----

def normalize_map01_to_pm1(local: np.ndarray) -> np.ndarray:
    # input: (1, H, W) with values {0,1}; output: [-1,1]
    return (local * 2.0) - 1.0

def normalize_goal_xy_to_pm1(gx: float, gy: float, W: int, H: int) -> Tuple[float, float]:
    # map pixel coords -> [-1,1]
    # careful: x in [0..W-1], y in [0..H-1]
    x_n = (gx / max(1, (W - 1))) * 2.0 - 1.0
    y_n = (gy / max(1, (H - 1))) * 2.0 - 1.0
    return x_n, y_n

def denorm_goal_pm1_to_xy(gx_n: float, gy_n: float, W: int, H: int) -> Tuple[float, float]:
    # not used now, but provided for completeness
    gx = (gx_n + 1.0) * 0.5 * (W - 1)
    gy = (gy_n + 1.0) * 0.5 * (H - 1)
    return gx, gy

def normalize_actions_to_pm1(actions: np.ndarray, max_speed: float, max_steer: float) -> np.ndarray:
    # actions: (..., 2) -> [v, steer]
    out = actions.copy()
    out[..., 0] = np.clip(out[..., 0] / max_speed, -1.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] / max_steer, -1.0, 1.0)
    return out

def denorm_actions_from_pm1(actions_n: torch.Tensor, max_speed: float, max_steer: float) -> torch.Tensor:
    # actions_n: (B, T, 2) in [-1,1]
    a = actions_n.contiguous()
    scale = torch.tensor([max_speed, max_steer], dtype=a.dtype, device=a.device).view(1, 1, 2)
    return torch.clamp(a, -1.0, 1.0) * scale
    #out = actions_n.clone()
    #out[..., 0] = out[..., 0].clamp(-1, 1) * max_speed
    #out[..., 1] = out[..., 1].clamp(-1, 1) * max_steer
    #return out

def ensure_nchw_single_channel(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is (B, 1, H, W).
    Accepts (B, H, W) -> unsqueeze channel.
    Accepts (B, H, W, 1) -> permute to (B, 1, H, W).
    If channels is wrong (e.g., 64), try to fix common mistakes.
    """
    if x.dim() == 3:
        # (B, H, W) -> (B, 1, H, W)
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        B, A, H, W = x.shape
        # If looks like NHWC (B, H, W, 1) or (B, H, W, C)
        if A not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Re-read after potential permute
        B, C, H, W = x.shape
        if C != 1:
            # Try to reduce to single channel if it looks like a stack gone wrong
            # Heuristic: if C equals H or W or batch-size “leaked”, keep only the first channel.
            x = x[:, :1, :, :]
    else:
        raise ValueError(f"local_map must be rank-3/4, got shape {tuple(x.shape)}")
    return x


# ----------------------------- Dataset -----------------------------

class NpzMapDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 history_size: int,
                 pred_horizon: int,
                 map_crop_size: int,
                 obs_dim: int = 6):
        self.files = sorted([p for p in glob.glob(os.path.join(root_dir, "*.npz"))])
        self.history_size = history_size
        self.pred_horizon = pred_horizon
        self.map_crop_size = map_crop_size
        self.obs_dim = obs_dim
        self._index: List[Tuple[int, str, int]] = []
        self._scan_files()

    def _scan_files(self):
        for fi, path in enumerate(self.files):
            try:
                with np.load(path, allow_pickle=True) as d:
                    keys = [k for k in d.files if k.endswith("path") or "suboptimal_path" in k or "optimal_path" in k]
                    for k in keys:
                        arr = d[k]  # (T, 6)
                        T = arr.shape[0]
                        max_t0 = T - (self.history_size + self.pred_horizon)
                        if max_t0 <= 0:
                            continue
                        for t0 in range(0, max_t0, 1):
                            self._index.append((fi, k, t0))
            except Exception as e:
                print(f"[WARN] Failed to index {path}: {e}")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i: int):
        fi, key, t0 = self._index[i]
        path = self.files[fi]
        with np.load(path, allow_pickle=True) as d:
            grid: np.ndarray = d["grid"].astype(np.uint8)
            goal: np.ndarray = d["goal_pos"].astype(np.float32)  # (y, x)
            traj: np.ndarray = d[key].astype(np.float32)         # (T, 6)

        hist = traj[t0: t0 + self.history_size]  # (H, 6)
        future = traj[t0 + self.history_size: t0 + self.history_size + self.pred_horizon]
        actions = future[:, [3, 5]]  # (P, 2)
        action_history = hist[:, [3, 5]]  # (H, 2)  last H actions

        cx, cy = float(hist[-1, 0]), float(hist[-1, 1])
        local = crop_local_map(grid, cx, cy, self.map_crop_size)  # (1, C, C)
        # action history & future actions to [-1,1]
        action_history_n = normalize_actions_to_pm1(action_history, DEFAULT_CONFIG["max_speed"], DEFAULT_CONFIG["max_steer"])
        actions_n = normalize_actions_to_pm1(actions, DEFAULT_CONFIG["max_speed"], DEFAULT_CONFIG["max_steer"])

        H, W = grid.shape
        gx, gy = float(goal[1]), float(goal[0])
        gx_n, gy_n = normalize_goal_xy_to_pm1(gx, gy, W, H) # goal to [-1,1]

        sample = {
            "action_history": torch.from_numpy(action_history_n).float(),  # (H,2) normalized
            "actions": torch.from_numpy(actions_n).float(),  # (P,2) normalized labels for diffusion
            "local_map": torch.from_numpy(local).float(),  # (1,C,C) already in [-1,1]
            "goal": torch.tensor([gx_n, gy_n], dtype=torch.float32),  # (2,) normalized
            "last_state": torch.from_numpy(hist[-1]),  # (6,) physical (keep as-is for rollout)
            "future_states": torch.from_numpy(future[:, :3])  # (P,3) physical (rollout target)
        }

        return sample

# ----------------------------- EMA -----------------------------

class EMAModel:
    """
    Exponential Moving Average for model parameters.
    Keeps a shadow copy updated as:
        ema = decay * ema + (1 - decay) * param
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().to(device if device is not None else p.device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow, f"EMA missing param: {name}"
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]):
        self.decay = float(state["decay"])
        self.shadow = {k: v for k, v in state["shadow"].items()}

# ----------------------------- Model -----------------------------

class LocalMapEncoder(nn.Module):
    def __init__(self, crop: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# RNN History Encoder
class ActionHistoryEncoder(nn.Module):
    def __init__(self, action_dim: int = 2, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=action_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.out_dim = hidden_size

    def forward(self, action_seq):  # (B, H, 2)
        # return last hidden state as embedding (B, hidden)
        _, h_n = self.gru(action_seq)   # h_n: (num_layers, B, hidden)
        h = h_n[-1]                     # (B, hidden)
        return h


class ConditionalDiffusionModel(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 history_size: int,
                 pred_horizon: int,
                 map_crop: int,
                 cond_dim: int = 128,
                 time_dim: int = 64,
                 hidden: int = 256,
                 action_dim: int = 2):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        self.map_enc = LocalMapEncoder(map_crop, out_dim=cond_dim)
        #self.hist_enc = HistoryEncoder(obs_dim, history_size, out_dim=cond_dim)
        self.hist_enc = ActionHistoryEncoder(action_dim=action_dim, hidden_size=cond_dim, num_layers=1)

        self.goal_fc = nn.Sequential(nn.Linear(2, cond_dim), nn.ReLU())
        self.t_fc = nn.Sequential(nn.Linear(time_dim, cond_dim), nn.ReLU())

        fused_dim = cond_dim * 4
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        seq_dim = pred_horizon * action_dim
        self.action_in = nn.Linear(seq_dim, hidden)
        self.out = nn.Linear(hidden, seq_dim)

        self.time_dim = time_dim

    def forward(self, noisy_actions, history, goal, local_map, t):
        m = self.map_enc(local_map)
        h = self.hist_enc(history)
        g = self.goal_fc(goal)
        te = self.t_fc(sinusoidal_time_embedding(t, self.time_dim))
        cond = torch.cat([m, h, g, te], dim=-1)
        cond = self.fuse(cond)
        a = self.action_in(noisy_actions)
        x = F.relu(a + cond)
        eps_hat = self.out(x)
        return eps_hat


# ----------------------------- Diffusion Core -----------------------------

class DiffusionScheduler:
    def __init__(self, T: int, beta_start: float, beta_end: float, device: str):
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_om * noise, noise

    def p_sample_step(self, model, x_t, t, cond_kwargs):
        beta_t = self.betas[t].view(-1, 1)
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar_t)
        eps_hat = model(x_t, **cond_kwargs, t=t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_ab) * eps_hat)
        if (t > 0).any():
            z = torch.randn_like(x_t)
            nonzero = (t > 0).float().view(-1, 1)
            sigma_t = torch.sqrt(beta_t)
            x_prev = mean + sigma_t * z * nonzero
        else:
            x_prev = mean
        return x_prev


# ----------------------------- Training Loop -----------------------------

def build_dataloaders(cfg: Dict):
    train_ds = NpzMapDataset(
        cfg["train_data_dir"],
        history_size=cfg["history_size"],
        pred_horizon=cfg["pred_horizon"],
        map_crop_size=cfg["map_crop_size"],
        obs_dim=cfg["obs_dim"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True
    )
    return train_loader


def build_model_and_sched(cfg: Dict):
    device = cfg["device"]
    model = ConditionalDiffusionModel(
        obs_dim=cfg["obs_dim"],
        history_size=cfg["history_size"],
        pred_horizon=cfg["pred_horizon"],
        map_crop=cfg["map_crop_size"],
        action_dim=cfg["action_dim"]
    ).to(device)
    sched = DiffusionScheduler(
        T=cfg["diffusion_T"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        device=device
    )
    return model, sched

def rollout_states_torch(x0, actions, dt=0.2, wheelbase=1.0,
                         max_steer=None, max_speed=None):
    """
    Differentiable bicycle-model rollout.

    x0:      (B, 6) tensor; we use at least [x, y, yaw] from it
    actions: (B, T, 2) -> [v, steer] over T steps
    Returns: (B, T, 3) -> [x, y, yaw] for each future step
    """
    B, T, _ = actions.shape
    x = x0[:, :3]  # [x, y, yaw]
    out = []

    v = actions[:, :, 0]
    s = actions[:, :, 1]
    if max_steer is not None:
        s = torch.clamp(s, -max_steer, max_steer)
    if max_speed is not None:
        v = torch.clamp(v, -max_speed, max_speed)

    for t in range(T):
        yaw = x[:, 2]
        x_next  = x[:, 0] + v[:, t] * torch.cos(yaw) * dt
        y_next  = x[:, 1] + v[:, t] * torch.sin(yaw) * dt
        yaw_next = yaw + (v[:, t] / wheelbase) * torch.tan(s[:, t]) * dt
        x = torch.stack([x_next, y_next, yaw_next], dim=1)
        out.append(x)

    return torch.stack(out, dim=1)  # (B, T, 3)


def train(cfg: Dict) -> None:
    """
    Train the conditional diffusion model with rollout loss and EMA.
    - Uses AMP (optional), per-step cosine LR schedule with warmup,
      gradient clipping, and EMA updates.
    - Saves checkpoint with {"model", "cfg", "ema"}.
    """
    set_seed(cfg["seed"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    device = cfg["device"]
    model, sched = build_model_and_sched(cfg)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # Build data loader BEFORE computing steps_per_epoch/total_steps
    loader = build_dataloaders(cfg)
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * max(1, int(cfg["epochs"]))
    warmup = int(0.05 * total_steps)  # 5% warmup

    def cosine_lr(step):
        # Linear warmup to 1.0, then cosine decay from 1.0 -> 0.1
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_lr)

    # AMP + EMA
    use_amp = bool(cfg.get("amp", True) and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_grad = float(cfg.get("max_grad_norm", 1.0))
    ema = EMAModel(model, decay=cfg.get("ema_decay", 0.999), device=device)

    global_step = 0
    model.train()
    for epoch in range(int(cfg["epochs"])):
        running = 0.0

        for batch in loader:
            non_block = bool(str(device).startswith("cuda"))
            act_hist = batch["action_history"].to(device, non_blocking=non_block)  # (B,H,2) normalized [-1,1]
            acts = batch["actions"].to(device, non_blocking=non_block)             # (B,P,2) normalized [-1,1]
            fut_states = batch["future_states"].to(device, non_blocking=non_block) # (B,P,3) physical
            local = ensure_nchw_single_channel(batch["local_map"].to(device, non_blocking=non_block))  # (B,1,C,C) in [-1,1]
            goal = batch["goal"].to(device, non_blocking=non_block)                # (B,2) normalized [-1,1]
            last_st = batch["last_state"].to(device, non_blocking=non_block)       # (B,6) physical

            B, P, D = acts.shape
            x0_actions = acts.reshape(B, P * D).contiguous()  # (B, P*2)

            # Sample diffusion time and corrupt actions to x_t
            t = torch.randint(0, sched.T, (B,), device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                x_t, noise = sched.q_sample(x0_actions, t)

                # Predict noise (standard DDPM training)
                eps_hat = model(
                    noisy_actions=x_t,
                    history=act_hist,
                    goal=goal,
                    local_map=local,
                    t=t
                )

                # 1) diffusion ε-loss
                loss_eps = F.mse_loss(eps_hat, noise)

                # 2) reconstruct clean actions and denormalize to physical space
                alpha_bar_t = sched.alphas_cumprod[t].unsqueeze(-1)        # (B,1)
                sqrt_ab = torch.sqrt(alpha_bar_t)
                sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
                x0_hat = (x_t - sqrt_one_minus * eps_hat) / (sqrt_ab + 1e-8)  # (B,P*2)

                pred_actions_norm = x0_hat.view(B, P, D).contiguous()  # normalized [-1,1]
                pred_actions_seq = denorm_actions_from_pm1(
                    pred_actions_norm, cfg["max_speed"], cfg["max_steer"]
                )  # (B,P,2) physical

                # 3) rollout predicted states and compare to GT future states
                pred_states = rollout_states_torch(
                    last_st, pred_actions_seq,
                    dt=cfg["dt"],
                    wheelbase=cfg["wheelbase"],
                    max_steer=cfg["max_steer"],
                    max_speed=cfg["max_speed"]
                )  # (B,P,3)

                pos_loss = F.mse_loss(pred_states[:, :, :2], fut_states[:, :, :2])
                yaw_loss = F.mse_loss(pred_states[:, :, 2], fut_states[:, :, 2])
                loss_roll = pos_loss + cfg["rollout_yaw_weight"] * yaw_loss

                # 4) total loss
                loss = loss_eps + cfg["rollout_weight"] * loss_roll

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Proper grad clipping with AMP
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)

            scaler.step(opt)
            scaler.update()

            # Per-step LR schedule (after optimizer step)
            scheduler.step()

            # EMA update AFTER optimizer step
            ema.update(model)

            running += float(loss.item())
            global_step += 1

        avg = running / max(1, steps_per_epoch)
        print(f"[Epoch {epoch+1}/{cfg['epochs']}] loss={avg:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint (optionally with EMA)
    ckpt_path = os.path.join(cfg["ckpt_dir"], cfg["model_name"])
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "ema": ema.state_dict(),
        },
        ckpt_path
    )
    print(f"Saved checkpoint: {ckpt_path}")


# ----------------------------- Environment -----------------------------

@dataclass
class CarState:
    x: float
    y: float
    yaw: float
    v: float
    throttle: float
    steer: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.v, self.throttle, self.steer], dtype=np.float32)


class GridCarEnv:
    def __init__(self, grid: np.ndarray, dt: float = 0.2, wheelbase: float = 1.0,
                 max_speed: float = 2.0, max_steer: float = 0.6):
        self.grid = grid.astype(np.uint8)
        self.h, self.w = grid.shape
        self.dt = dt
        self.L = wheelbase
        self.max_speed = max_speed
        self.max_steer = max_steer

    def in_bounds(self, x: float, y: float) -> bool:
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < self.w and 0 <= yi < self.h

    def collision(self, x: float, y: float) -> bool:
        if not self.in_bounds(x, y):
            return True
        xi, yi = int(round(x)), int(round(y))
        return self.grid[yi, xi] == 1

    def step(self, state: CarState, action: Tuple[float, float]) -> Tuple[CarState, bool]:
        v_cmd = float(np.clip(action[0], -self.max_speed, self.max_speed))
        steer = float(np.clip(action[1], -self.max_steer, self.max_steer))

        x, y, yaw, v = state.x, state.y, state.yaw, v_cmd
        x_next = x + v * math.cos(yaw) * self.dt
        y_next = y + v * math.sin(yaw) * self.dt
        yaw_next = yaw + (v / self.L) * math.tan(steer) * self.dt

        ns = CarState(x_next, y_next, yaw_next, v, 0.0, steer)
        collided = self.collision(ns.x, ns.y)
        return ns, collided

    def sample_local_start_goal(grid: np.ndarray,
                                max_dist: int = 5,
                                try_start_from_given: Optional[Tuple[int, int]] = None,
                                max_tries: int = 200) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Pick start/goal such that both are free cells and distance(start, goal) <= max_dist.
        If try_start_from_given provided, use it as start (if free); otherwise sample start at random.
        Uses Manhattan distance; change to Euclidean if desired.
        Returns: ((sy, sx), (gy, gx))
        """
        H, W = grid.shape
        rng = np.random.default_rng()

        # choose start
        if try_start_from_given is not None:
            sy, sx = try_start_from_given
            if sy < 0 or sy >= H or sx < 0 or sx >= W or grid[sy, sx] == 1:
                try_start_from_given = None
        if try_start_from_given is None:
            for _ in range(max_tries):
                sy, sx = int(rng.integers(0, H)), int(rng.integers(0, W))
                if grid[sy, sx] == 0:  # free
                    break
            else:
                raise RuntimeError("Failed to sample a free START cell")

        if try_start_from_given is not None:
            sy, sx = try_start_from_given

        # choose goal within max_dist (Manhattan)
        for _ in range(max_tries):
            dy = int(rng.integers(-max_dist, max_dist + 1))
            rem = max_dist - abs(dy)
            dx = int(rng.integers(-rem, rem + 1))
            gy, gx = sy + dy, sx + dx
            if 0 <= gy < H and 0 <= gx < W and grid[gy, gx] == 0:
                if abs(dy) + abs(dx) <= max_dist:
                    return (sy, sx), (gy, gx)

        # fallback: scan a local window
        for dy in range(-max_dist, max_dist + 1):
            for dx in range(-max_dist + abs(dy), max_dist - abs(dy) + 1):
                gy, gx = sy + dy, sx + dx
                if 0 <= gy < H and 0 <= gx < W and grid[gy, gx] == 0:
                    return (sy, sx), (gy, gx)

        raise RuntimeError("Failed to sample a GOAL within max_dist from START")


# ----------------------------- DiTree Planner -----------------------------

class Node:
    def __init__(self, state: CarState, parent: Optional['Node'] = None,
                 path_from_parent: Optional[np.ndarray] = None,
                 actions_from_parent: Optional[np.ndarray] = None):
        self.state = state
        self.parent = parent
        self.path_from_parent = path_from_parent  # (K, 6)
        self.actions_from_parent = actions_from_parent  # (K-1, 2) actions used to get here
        self.children: List['Node'] = []

class DiffusionTreePlanner:
    def __init__(self,
                 env: GridCarEnv,
                 model: ConditionalDiffusionModel,
                 sched: DiffusionScheduler,
                 cfg: Dict):
        self.env = env
        self.model = model
        self.sched = sched
        self.cfg = cfg
        self.device = cfg["device"]
        self.grid_torch = torch.from_numpy(self.env.grid.astype(np.float32)).to(self.device)

    def build_tree(self,
                   start: Tuple[int, int],
                   goal: Tuple[int, int],
                   max_iterations: Optional[int] = None) -> Tuple[List[Node], Optional[Node]]:
        if max_iterations is None:
            max_iterations = self.cfg["max_iterations"]

        gx, gy = float(goal[1]), float(goal[0])
        sx, sy = float(start[1]), float(start[0])
        yaw0 = math.atan2(gy - sy, gx - sx)
        root = Node(CarState(sx, sy, yaw0, 0.0, 0.0, 0.0), parent=None, path_from_parent=None)
        nodes: List[Node] = [root]
        goal_node: Optional[Node] = None

        self.model.eval()
        for _ in range(max_iterations):
            node = self._select_node(nodes, (gx, gy))
            H = self.cfg["history_size"]
            act_hist = self._collect_action_history(node, H)  # (H,2)
            actions = self.sample_actions_k(node, goal=(gx, gy), act_hist=act_hist)
            traj_states, valid = self._simulate_segment(node.state, actions)
            if not valid:
                continue

            new_state = CarState(*traj_states[-1])
            new_node = Node(new_state, parent=node, path_from_parent=traj_states, actions_from_parent=actions)   # (A,2)
            node.children.append(new_node)
            nodes.append(new_node)

            if math.hypot(new_state.x - gx, new_state.y - gy) <= self.cfg["goal_tolerance"]:
                goal_node = new_node
                break

        return nodes, goal_node

    def _select_node(self, nodes, goal_xy):
        gx, gy = goal_xy
        scores = []
        for n in nodes:
            d = math.hypot(n.state.x - gx, n.state.y - gy)
            s = 1.0 / (1e-3 + d) + 0.01 * depth(n)  # קרוב יותר + עמוק יותר
            scores.append(s)
        probs = np.array(scores, dtype=np.float64)
        probs /= probs.sum()
        idx = np.random.choice(len(nodes), p=probs)
        return nodes[idx]

    def _collect_history(self, node: Node, limit: int) -> List[np.ndarray]:
        seq: List[np.ndarray] = []
        cur = node
        while cur is not None and len(seq) < limit:
            if cur.path_from_parent is None:
                seq.append(cur.state.to_array())
            else:
                for s in reversed(cur.path_from_parent):
                    seq.append(s)
                    if len(seq) >= limit:
                        break
            cur = cur.parent
        seq = list(reversed(seq))
        if len(seq) < limit:
            pad = [seq[0]] * (limit - len(seq))
            seq = pad + seq
        return seq[-limit:]

    def _collect_action_history(self, node: 'Node', H: int) -> np.ndarray:
        seq = []
        cur = node
        while cur is not None and len(seq) < H:
            if cur.actions_from_parent is not None and len(cur.actions_from_parent) > 0:
                for a in reversed(cur.actions_from_parent):
                    seq.append(a)
                    if len(seq) >= H:
                        break
            # for root: no actions_from_parent -> pad later
            cur = cur.parent
        seq = list(reversed(seq))
        if len(seq) < H:
            pad = [np.zeros((2,), dtype=np.float32)] * (H - len(seq))
            seq = pad + seq
        return np.stack(seq[-H:], axis=0)  # (H,2)

    @torch.no_grad()
    def sample_actions(self, node: 'Node', goal: Tuple[float, float],
                       act_hist: Optional[np.ndarray] = None) -> np.ndarray:
        H = self.cfg["history_size"];
        P = self.cfg["pred_horizon"]
        A = self.cfg["action_horizon"];
        D = self.cfg["action_dim"]
        crop = self.cfg["map_crop_size"];
        device = self.device
        max_speed = self.env.max_speed;
        max_steer = self.env.max_steer
        Hh, Ww = self.env.h, self.env.w

        # 1) local crop (already set to [-1,1] by crop_local_map_torch)
        cx, cy = float(node.state.x), float(node.state.y)
        local_t = ensure_nchw_single_channel(crop_local_map_torch(self.grid_torch, cx, cy, crop).unsqueeze(0).to(device))

        # 2) action history
        if act_hist is None:
            act_hist = self._collect_action_history(node, H)  # (H,2) physical
        # normalize to [-1,1]
        act_hist_n = normalize_actions_to_pm1(act_hist, max_speed, max_steer)
        mode = self.cfg.get("history_mode", "full")
        if mode == "last":
            act_hist_n = np.tile(act_hist_n[-1:], (H, 1))
        elif mode == "none":
            act_hist_n = np.zeros_like(act_hist_n)
        hist_t = torch.from_numpy(act_hist_n.astype(np.float32)).unsqueeze(0).to(device)

        # 3) goal normalized to [-1,1]
        gx_n, gy_n = normalize_goal_xy_to_pm1(goal[0], goal[1], Ww, Hh)
        goal_t = torch.tensor([[gx_n, gy_n]], dtype=torch.float32, device=device)

        # 4) reverse diffusion in normalized action space
        seq_dim = P * D
        x_t = torch.randn(1, seq_dim, device=device)
        for t_step in reversed(range(self.sched.T)):
            t = torch.tensor([t_step], device=device, dtype=torch.long)
            cond = dict(history=hist_t, goal=goal_t, local_map=local_t)
            x_t = self.sched.p_sample_step(self.model.forward, x_t, t, cond)

        # x0 normalized -> reshape -> DENORM to physical -> clamp for safety
        x0_norm = x_t.view(1, P, D)
        actions_phys = denorm_actions_from_pm1(x0_norm, max_speed, max_steer)
        actions_phys = actions_phys[:, :A, :].clamp_(
            torch.tensor([-max_speed, -max_steer], device=device),
            torch.tensor([max_speed, max_steer], device=device)
        )
        return actions_phys.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def sample_actions_k(
            self,
            node: "Node",
            goal: Tuple[float, float],
            act_hist: Optional[np.ndarray] = None,
            K: int = 4,
            step_penalty: float = 0.01,
    ) -> np.ndarray:
        """
        Propose K candidate action sequences via diffusion and score them using the
        SAME segment simulator used for tree expansion (_simulate_segment), so that
        collision handling and sub-step checks are consistent.

        Scoring (higher is better):
            score = (d_start_to_goal - d_end_to_goal) - step_penalty * num_actions
        Candidates that collide (invalid segment) are discarded.

        If all K candidates collide, fall back to a single diffusion sample via sample_actions.
        """
        gx, gy = float(goal[0]), float(goal[1])
        x0, y0 = float(node.state.x), float(node.state.y)
        best_score = -float("inf")
        best_actions: Optional[np.ndarray] = None

        # Try K candidates sampled by diffusion
        for _ in range(max(1, K)):
            actions = self.sample_actions(node, goal, act_hist)  # (A, 2) in physical units
            # Evaluate this candidate with the *same* segment simulator used in build_tree:
            traj_states, valid = self._simulate_segment(node.state, actions)
            if not valid or traj_states.shape[0] == 0:
                continue  # discard colliding/invalid candidate

            # End state after executing the whole segment:
            x1 = float(traj_states[-1, 0])
            y1 = float(traj_states[-1, 1])

            # Distance improvement towards the goal:
            d0 = math.hypot(x0 - gx, y0 - gy)
            d1 = math.hypot(x1 - gx, y1 - gy)
            score = (d0 - d1) - step_penalty * actions.shape[0]

            if score > best_score:
                best_score = score
                best_actions = actions

        # Fallback: if every candidate collided, just return a single diffusion sample
        if best_actions is None:
            best_actions = self.sample_actions(node, goal, act_hist)

        return best_actions

    def _simulate_segment(self, start: CarState, actions: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Simulate one expansion segment with robust collision checking:
        - Uses sub-sampling along each motion to avoid skipping obstacles between steps.
        - Sub-sample spacing controlled by cfg['collision_substep_max_dist'] (in grid cells).

        Returns:
            (states, valid)
            states: (K, 6) including the start state; K = len(actions)+1 if valid, shorter on early collision
            valid:  True if no collision/out-of-bounds encountered, False otherwise.
        """
        states = [start.to_array()]
        cur = start

        max_substep = float(self.cfg.get("collision_substep_max_dist", 0.5))
        max_substep = max(1e-6, max_substep)  # guard against zero/negatives

        for a in actions:
            # Predict next state using the same single-step model as before:
            nxt, _ = self.env.step(cur, (float(a[0]), float(a[1])))

            # Sub-sample along the straight line (x,y) from cur -> nxt
            dx = nxt.x - cur.x
            dy = nxt.y - cur.y
            dist = math.hypot(dx, dy)
            n_sub = max(1, int(math.ceil(dist / max_substep)))

            collided = False
            for k in range(1, n_sub + 1):
                alpha = k / n_sub
                xk = cur.x + alpha * dx
                yk = cur.y + alpha * dy
                if self.env.collision(xk, yk):
                    collided = True
                    break

            if collided:
                # Early return: keep partial states to aid debugging/visualization
                return np.stack(states, axis=0), False

            # Accept this step
            cur = nxt
            states.append(cur.to_array())

        return np.stack(states, axis=0), True


# ----------------------------- Inference / Evaluation -----------------------------

def load_checkpoint(ckpt_path: str, cfg: Dict) -> Tuple[ConditionalDiffusionModel, DiffusionScheduler]:
    ckpt = torch.load(ckpt_path, map_location=cfg["device"])
    saved_cfg = ckpt.get("cfg", cfg)
    merged = {**saved_cfg, **cfg}

    model, sched = build_model_and_sched(merged)
    model.load_state_dict(ckpt["model"])

    if merged.get("use_ema_for_eval", True) and ("ema" in ckpt):
        ema_tmp = EMAModel(model, decay=merged.get("ema_decay", 0.999), device=cfg["device"])
        ema_tmp.load_state_dict(ckpt["ema"])
        ema_tmp.copy_to(model)

    model.eval()
    return model, sched



def extract_best_path(goal_node: Optional[Node], nodes: List[Node], goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
    # choose chain to goal if found; otherwise, closest-to-goal node
    def node_chain(n: Node) -> List[Node]:
        chain = []
        cur = n
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        return list(reversed(chain))

    if goal_node is None:
        gx, gy = goal_xy
        best = None
        best_d = 1e9
        for n in nodes:
            d = math.hypot(n.state.x - gx, n.state.y - gy)
            if d < best_d:
                best_d = d
                best = n
        chain = node_chain(best)
    else:
        chain = node_chain(goal_node)

    # stitch states along chain edges (avoid duplicating nodes)
    path_xy: List[Tuple[float, float]] = []
    for i, n in enumerate(chain):
        if i == 0:
            path_xy.append((n.state.x, n.state.y))
        else:
            edge_traj = n.path_from_parent  # (K, 6)
            if edge_traj is None or len(edge_traj) == 0:
                path_xy.append((n.state.x, n.state.y))
            else:
                for s in edge_traj[1:]:  # skip first to avoid duplicate
                    path_xy.append((float(s[0]), float(s[1])))
    return path_xy

def _node_chain(n: 'Node') -> List['Node']:
    chain = []
    cur = n
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    return list(reversed(chain))

def extract_path_xy_from_node(n: 'Node') -> List[Tuple[float, float]]:
    """Stitch (x,y) along the chain root→n without duplicating joints."""
    chain = _node_chain(n)
    path_xy: List[Tuple[float, float]] = []
    for i, nd in enumerate(chain):
        if i == 0:
            path_xy.append((nd.state.x, nd.state.y))
        else:
            edge_traj = nd.path_from_parent
            if edge_traj is None or len(edge_traj) == 0:
                path_xy.append((nd.state.x, nd.state.y))
            else:
                for s in edge_traj[1:]:
                    path_xy.append((float(s[0]), float(s[1])))
    return path_xy

def topk_candidate_nodes(nodes: List['Node'],
                         goal_xy: Tuple[float, float],
                         k: int = 5) -> List['Node']:
    """Return up to K distinct candidates: the goal node (if any) + nearest leaves/late nodes."""
    gx, gy = goal_xy
    # Prefer leaves (no children). If tree is shallow, also allow late nodes.
    leaves = [n for n in nodes if len(n.children) == 0]
    pool = leaves if leaves else nodes

    scored = []
    for n in pool:
        d = math.hypot(n.state.x - gx, n.state.y - gy)
        scored.append((d, -depth(n), n))
    # Tie-break by deeper = better (-depth)
    scored.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in scored[:max(1, k)]]

def depth(n: 'Node') -> int:
    d = 0
    cur = n
    while cur is not None:
        d += 1
        cur = cur.parent
    return d - 1


def collect_edges(nodes: List[Node]) -> List[List[Tuple[float, float]]]:
    edges: List[List[Tuple[float, float]]] = []
    for n in nodes:
        if n.parent is None or n.path_from_parent is None:
            continue
        pts = [(float(s[0]), float(s[1])) for s in n.path_from_parent]
        edges.append(pts)
    return edges


def visualize_map_and_paths(
    grid: np.ndarray,
    edges: List[List[Tuple[float, float]]],
    paths: List[List[Tuple[float, float]]],
    out_png: str,
    start_xy: Tuple[float, float] = None,
    goal_xy: Tuple[float, float] = None,
    path_labels: Optional[List[str]] = None,
):
    """
    Save a PNG with the map, full tree edges, and multiple candidate paths.
    - paths: list of polylines (each list[(x,y)])
    - path_labels: optional labels per path (same length as paths)
    """
    # Map colors like the reference image: free = light gray, obstacle = black
    cmap = ListedColormap(["#d3d3d3", "#000000"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.astype(np.uint8), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    # All tree edges (light blue)
    edge_lines = []
    for e in edges:
        if len(e) < 2:
            continue
        xs = [p[0] for p in e]; ys = [p[1] for p in e]
        line, = ax.plot(xs, ys, linewidth=0.8, alpha=0.6, color="#3b82f6", label="_nolegend_")
        edge_lines.append(line)

    # Candidate paths (distinct colors)
    palette = ["#ef4444", "#10b981", "#8b5cf6", "#f59e0b", "#06b6d4", "#e11d48", "#84cc16"]
    handles, labels = [], []
    if edge_lines:
        handles.append(edge_lines[0]); labels.append("Tree edges")

    for i, path in enumerate(paths):
        if not path or len(path) < 2:
            continue
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        color = palette[i % len(palette)]
        lab = path_labels[i] if (path_labels and i < len(path_labels)) else (f"Path {i+1}")
        h, = ax.plot(xs, ys, linewidth=2.2 if i == 0 else 1.8, color=color, label=lab)
        handles.append(h); labels.append(lab)

    # Start / Goal markers
    if start_xy is not None:
        s = ax.scatter([start_xy[0]], [start_xy[1]], s=60, marker="*", color="#22c55e",
                       edgecolor="black", linewidths=0.6, label="Start")
        handles.append(s); labels.append("Start")
    if goal_xy is not None:
        g = ax.scatter([goal_xy[0]], [goal_xy[1]], s=60, marker="*", color="#f59e0b",
                       edgecolor="black", linewidths=0.6, label="Goal")
        handles.append(g); labels.append("Goal")

    # Axes formatting
    ax.set_xlim([-0.5, grid.shape[1]-0.5])
    ax.set_ylim([grid.shape[0]-0.5, -0.5])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("DiTree — Tree and Candidate Paths")

    # Put a legend OUTSIDE the map (right side)
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0, frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)




def evaluate_over_testset(cfg: Dict, out_dir: str, visualize: bool = True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    #test_files = sorted(glob.glob(os.path.join(cfg["test_data_dir"], "*.npz")))
    test_files = [f for f in glob.glob(os.path.join(cfg["test_data_dir"], "*.npz"))if "grid" in np.load(f, allow_pickle=True).files]
    if not test_files:
        print(f"No test files found in {cfg['test_data_dir']}")
        return ""

    ckpt_path = os.path.join(cfg["ckpt_dir"], cfg["model_name"])
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Train first to enable inference.")
        return ""

    # load model + planner (reinitialized per grid size)
    all_rows = []
    for path in test_files:
        with np.load(path, allow_pickle=True) as d:
            print(f"Evaluating {os.path.basename(path)}…")
            grid = d["grid"].astype(np.uint8)
            start = tuple(map(int, d["start_pos"].tolist()))   # (y,x)
            goal  = tuple(map(int, d["goal_pos"].tolist()))    # (y,x)
            r_max = cfg.get("goal_radius")
            r_min = cfg.get("goal_radius_min")

            if cfg.get("random_pair", False):
                # Random free START, then GOAL inside radius
                H, W = grid.shape
                rng = np.random.default_rng()
                for _ in range(500):
                    sy, sx = int(rng.integers(0, H)), int(rng.integers(0, W))
                    if grid[sy, sx] == 0:
                        start = (sy, sx)
                        break
                else:
                    raise RuntimeError("Failed to sample a free START cell for random_pair")

                goal = sample_goal_within_radius(grid, start, r_max=r_max, r_min=r_min)

            elif cfg.get("random_local_goal", False):
                # Keep dataset START, sample GOAL inside radius
                goal = sample_goal_within_radius(grid, start, r_max=r_max, r_min=r_min)

            elif cfg.get("force_radius", False):
                # If dataset goal is too far/too close, re-sample it to satisfy the annulus
                d_now = euclidean_distance(start, goal)
                if not (r_min <= d_now <= r_max):
                    goal = sample_goal_within_radius(grid, start, r_max=r_max, r_min=r_min)

            # For transparency/debugging:
            dist_used = euclidean_distance(start, goal)
            print(f"Using start={start}, goal={goal}, euclid_distance={dist_used:.2f} (radius <= {r_max})")

        env = GridCarEnv(grid, dt=cfg["dt"], wheelbase=cfg["wheelbase"],
                         max_speed=cfg["max_speed"], max_steer=cfg["max_steer"])

        model, sched = load_checkpoint(ckpt_path, cfg)
        planner = DiffusionTreePlanner(env, model, sched, cfg)

        t0 = time.perf_counter()
        nodes, goal_node = planner.build_tree(start, goal)
        runtime_sec = time.perf_counter() - t0

        reached = goal_node is not None
        edges = collect_edges(nodes)
        best_path = extract_best_path(goal_node, nodes, goal_xy=(goal[1], goal[0]))

        # Top-K candidate nodes & paths (K includes the best if it exists)
        K = int(cfg.get("viz_topk_paths", 5))
        cands = []
        labels = []
        if goal_node is not None:
            cands.append(goal_node)
            labels.append("Best (goal)")

        # add more close-to-goal leaves
        extra = topk_candidate_nodes(nodes, goal_xy=(goal[1], goal[0]), k=K)
        for n in extra:
            if n is not goal_node:  # avoid duplicates
                cands.append(n)
                labels.append(f"Candidate {len(cands)}")

        # build the polylines
        paths = [extract_path_xy_from_node(n) for n in cands]

        # actual end position:
        if best_path:
            end_xy = (float(best_path[-1][0]), float(best_path[-1][1]))
        else:
            # fallback: last node state
            last_node = nodes[-1] if nodes else None
            end_xy = (float(last_node.state.x), float(last_node.state.y)) if last_node else (float(start[1]), float(start[0]))

        # per-file output folder
        base = os.path.splitext(os.path.basename(path))[0]
        out_sub = os.path.join(out_dir, base)
        os.makedirs(out_sub, exist_ok=True)

        # save edges json
        edges_json = [ {"points": e} for e in edges ]
        with open(os.path.join(out_sub, "ditree_edges.json"), "w", encoding="utf-8") as f:
            json.dump(edges_json, f, indent=2)

        # save best path csv
        if best_path:
            bp_arr = np.array(best_path, dtype=np.float32)
            np.savetxt(os.path.join(out_sub, "best_path.csv"), bp_arr, delimiter=",", header="x,y", comments="")

        # save meta.json (start/goal/end/nodes/runtime)
        meta = {
            "file": os.path.basename(path),
            "start_yx": [int(start[0]), int(start[1])],
            "goal_yx":  [int(goal[0]), int(goal[1])],
            "end_xy":   [float(end_xy[0]), float(end_xy[1])],
            "nodes_expanded": int(len(nodes)),
            "reached_goal": int(reached),
            "best_path_len": int(len(best_path)),
            "runtime_sec": float(runtime_sec),
            "history_mode": cfg.get("history_mode", "full"),
            "euclid_start_goal": float(dist_used),
            "goal_radius_max": float(r_max),
            "goal_radius_min": float(r_min),
        }
        with open(os.path.join(out_sub, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # save visualization
        if visualize:
            visualize_map_and_paths(
                grid, edges, paths,
                out_png=os.path.join(out_sub, "snapshot_multi.png"),
                start_xy=(start[1], start[0]),
                goal_xy=(goal[1], goal[0]),
                path_labels=labels
            )

        row = {
            "file": os.path.basename(path),
            "nodes_expanded": len(nodes),
            "reached_goal": int(reached),
            "best_path_len": len(best_path),
            "start_yx": start,
            "goal_yx": goal,
            "end_x": end_xy[0],
            "end_y": end_xy[1],
            "runtime_sec": runtime_sec,
            "euclid_start_goal": dist_used,
            "goal_radius_min": r_min,
            "goal_radius_max": r_max,
        }
        all_rows.append(row)
        print(f"[TEST] {row}")

    # summary CSV
    summary_csv = os.path.join(out_dir, "summary.csv")
    import csv
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file","nodes_expanded","reached_goal","best_path_len","start_yx","goal_yx","end_x","end_y",
                      "runtime_sec","history_mode","euclid_start_goal","goal_radius_min","goal_radius_max"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            r["runtime_sec"] = float(r["runtime_sec"])
            r["history_mode"] = cfg.get("history_mode","full")
            r["reached_goal"] = int(r["reached_goal"])
            r["euclid_start_goal"] = float(dist_used)
            r["goal_radius_min"] = float(r_min)
            r["goal_radius_max"] = float(r_max)

            w.writerow(r)
    print(f"Wrote summary: {summary_csv}")
    return summary_csv

def run_eval_once(cfg: Dict, out_dir: str, visualize: bool = True) -> str:
    """
    Helper to run a single evaluation pass using current cfg['history_mode'].
    Returns path to summary.csv (or empty string on failure).
    """
    return evaluate_over_testset(cfg, out_dir=out_dir, visualize=visualize)

# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=0, help="Number of training epochs (0 = skip training).")
    parser.add_argument("--eval_all_tests", action="store_true", help="Run inference over ALL test NPZ files.")
    parser.add_argument("--visualize", action="store_true", help="Save PNG visualizations per test map.")
    parser.add_argument("--out_dir", type=str, default="outputs_test_history", help="Output directory for eval results.")
    parser.add_argument("--pipeline", action="store_true", help="Run: (optional) training -> eval NONE -> eval FULL in one command.")
    parser.add_argument("--out_dir_none", type=str, default="meta_outputs_test_no_history", help="Output dir for history_mode=none in pipeline/evals.")
    parser.add_argument("--out_dir_full", type=str, default="meta_outputs_test_history", help="Output dir for history_mode=full in pipeline/evals.")
    parser.add_argument("--history_mode", type=str, default="full", choices=["full","last","none"], help="Inference conditioning: full/last/none")
    parser.add_argument("--random_local_goal", action="store_true", help="Override dataset goal with a random free goal within <=5 cells from start.")
    parser.add_argument("--random_pair", action="store_true", help="Ignore dataset start/goal; sample BOTH start and goal with <=5 cells distance.")
    parser.add_argument("--max_local_dist", type=int, default=5, help="Max Manhattan distance between start and goal when randomizing.")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["history_mode"] = args.history_mode
    cfg["out_dir"] = args.out_dir
    cfg["epochs"] = args.epochs
    cfg["eval_all_tests"] = args.eval_all_tests
    cfg["visualize"] = args.visualize
    cfg["random_local_goal"] = bool(args.random_local_goal)
    cfg["random_pair"] = bool(args.random_pair)
    cfg["max_local_dist"] = int(args.max_local_dist)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    print("Using device:", cfg["device"])
    print("Train dir:", cfg["train_data_dir"])
    print("Test dir:", cfg["test_data_dir"])

    # ---- PIPELINE: train (optional) -> eval NONE -> eval FULL ----
    if args.pipeline:
        # 1) optional training
        if args.epochs > 0:
            if os.path.isdir(cfg["train_data_dir"]) and glob.glob(os.path.join(cfg["train_data_dir"], "*.npz")):
                print("[PIPELINE] Training ...")
                train(cfg)
            else:
                print(f"[PIPELINE] Training data not found under {cfg['train_data_dir']}. Skipping training.")
        else:
            print("[PIPELINE] Skipping training (epochs=0).")

        # 2) eval with NO HISTORY
        cfg_none = cfg.copy()
        cfg_none["history_mode"] = "none"
        print(f"[PIPELINE] Evaluating history_mode=none -> {args.out_dir_none}")
        run_eval_once(cfg_none, out_dir=args.out_dir_none, visualize=args.visualize)

        # 3) eval with FULL HISTORY
        cfg_full = cfg.copy()
        cfg_full["history_mode"] = "full"
        print(f"[PIPELINE] Evaluating history_mode=full -> {args.out_dir_full}")
        run_eval_once(cfg_full, out_dir=args.out_dir_full, visualize=args.visualize)

        print("[PIPELINE] Done.")
        return


    if args.epochs > 0:
        # train only if data exists
        if os.path.isdir(cfg["train_data_dir"]) and glob.glob(os.path.join(cfg["train_data_dir"], "*.npz")):
            print("Starting training…")
            train(cfg)
        else:
            print(f"Training data not found under {cfg['train_data_dir']}. Skipping training.")

    if args.eval_all_tests:
        evaluate_over_testset(cfg, out_dir=args.out_dir, visualize=args.visualize)
    else:
        # default single-file demo on first test if exists
        test_files = sorted(glob.glob(os.path.join(cfg["test_data_dir"], "*.npz")))
        if not test_files:
            print(f"No test NPZ files found in {cfg['test_data_dir']}.")
            return

        ckpt_path = os.path.join(cfg["ckpt_dir"], cfg["model_name"])
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found at {ckpt_path}. Train first to enable inference.")
            return

        with np.load(test_files[0], allow_pickle=True) as d:
            grid = d["grid"].astype(np.uint8)
            start = tuple(map(int, d["start_pos"].tolist()))
            goal = tuple(map(int, d["goal_pos"].tolist()))

        env = GridCarEnv(grid, dt=cfg["dt"], wheelbase=cfg["wheelbase"],
                         max_speed=cfg["max_speed"], max_steer=cfg["max_steer"])
        model, sched = load_checkpoint(ckpt_path, cfg)
        planner = DiffusionTreePlanner(env, model, sched, cfg)
        t0 = time.perf_counter()
        nodes, goal_node = planner.build_tree(start, goal)
        runtime_sec = time.perf_counter() - t0

        edges = collect_edges(nodes)
        best_path = extract_best_path(goal_node, nodes, goal_xy=(goal[1], goal[0]))

        # actual end position:
        if best_path:
            end_xy = (float(best_path[-1][0]), float(best_path[-1][1]))
        else:
            last_node = nodes[-1] if nodes else None
            end_xy = (float(last_node.state.x), float(last_node.state.y)) if last_node else (float(start[1]),
                                                                                             float(start[0]))

        demo_dir = os.path.join(args.out_dir, "demo_single")
        os.makedirs(demo_dir, exist_ok=True)

        # save best path CSV + edges JSON
        if best_path:
            np.savetxt(os.path.join(demo_dir, "best_path.csv"),
                       np.array(best_path, dtype=np.float32),
                       delimiter=",", header="x,y", comments="")
        with open(os.path.join(demo_dir, "ditree_edges.json"), "w", encoding="utf-8") as f:
            json.dump([{"points": e} for e in edges], f, indent=2)

        # save meta.json
        meta = {
            "file": os.path.basename(test_files[0]),
            "start_yx": [int(start[0]), int(start[1])],
            "goal_yx": [int(goal[0]), int(goal[1])],
            "end_xy": [float(end_xy[0]), float(end_xy[1])],
            "nodes_expanded": int(len(nodes)),
            "reached_goal": int(goal_node is not None),
            "best_path_len": int(len(best_path)),
            "runtime_sec": float(runtime_sec),
            "history_mode": cfg.get("history_mode", "full"),
        }
        with open(os.path.join(demo_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # visualization
        if args.visualize:
            paths, labels = [], []
            if best_path:
                paths.append(best_path);
                labels.append("Best (goal)" if goal_node is not None else "Best (closest)")
            visualize_map_and_paths(
                grid, edges, paths,
                out_png=os.path.join(demo_dir, "snapshot_multi.png"),
                start_xy=(start[1], start[0]),
                goal_xy=(goal[1], goal[0]),
                path_labels=labels
            )
        print(f"Single test demo saved under: {demo_dir}")


if __name__ == "__main__":
    main()
