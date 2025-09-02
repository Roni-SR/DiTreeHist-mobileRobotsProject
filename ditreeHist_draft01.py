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

    # training
    "batch_size": 64,
    "epochs": 0,
    "lr": 1e-3,
    "num_workers": 0,

    # diffusion
    "diffusion_T": 64,          # number of diffusion steps
    "beta_start": 1e-4,
    "beta_end": 0.02,

    # tree / env
    "dt": 0.2,                  # simulation step time (s)
    "wheelbase": 1.0,           # bicycle model L
    "action_horizon": 8,        # steps to execute when expanding a node (<= pred_horizon)
    "max_speed": 2.0,
    "max_steer": 0.6,           # ~34 degrees
    "goal_tolerance": 1.5,      # distance to goal in grid units to stop
    "max_iterations": 500,      # tree expansion iterations

    # checkpoints
    "ckpt_dir": "checkpoints",
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
        cut = np.pad(cut, (
            (0, max(0, crop - cut.shape[0])),
            (0, max(0, crop - cut.shape[1]))
        ), constant_values=1)
        cut = cut[:crop, :crop]
    return cut.astype(np.float32)[None, ...]


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

        hist = traj[t0: t0 + self.history_size]                 # (H, 6)
        future = traj[t0 + self.history_size: t0 + self.history_size + self.pred_horizon]
        actions = future[:, [3, 5]]                             # (P, 2)

        cx, cy = float(hist[-1, 0]), float(hist[-1, 1])
        local = crop_local_map(grid, cx, cy, self.map_crop_size)  # (1, C, C)

        gx, gy = float(goal[1]), float(goal[0])
        sample = {
            "history": torch.from_numpy(hist),
            "actions": torch.from_numpy(actions),
            "local_map": torch.from_numpy(local),
            "goal": torch.tensor([gx, gy], dtype=torch.float32)
        }
        return sample


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


class HistoryEncoder(nn.Module):
    def __init__(self, obs_dim: int, history_size: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * history_size, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.ReLU()
        )

    def forward(self, hist):
        b, h, d = hist.shape
        return self.net(hist.reshape(b, h * d))


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
        self.hist_enc = HistoryEncoder(obs_dim, history_size, out_dim=cond_dim)

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


def train(cfg: Dict):
    set_seed(cfg["seed"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    device = cfg["device"]
    model, sched = build_model_and_sched(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    loader = build_dataloaders(cfg)

    model.train()
    for epoch in range(cfg["epochs"]):
        running = 0.0
        for batch in loader:
            hist = batch["history"].to(device)
            acts = batch["actions"].to(device)
            local = batch["local_map"].to(device)
            goal = batch["goal"].to(device)

            B, P, D = acts.shape
            x0 = acts.reshape(B, P * D)

            t = torch.randint(0, sched.T, (B,), device=device)
            x_t, noise = sched.q_sample(x0, t)

            eps_hat = model(
                noisy_actions=x_t,
                history=hist,
                goal=goal,
                local_map=local,
                t=t
            )
            loss = F.mse_loss(eps_hat, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()

        avg = running / max(1, len(loader))
        print(f"[Epoch {epoch+1}/{cfg['epochs']}] loss={avg:.4f}")

    ckpt_path = os.path.join(cfg["ckpt_dir"], cfg["model_name"])
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
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


# ----------------------------- DiTree Planner -----------------------------

class Node:
    def __init__(self, state: CarState, parent: Optional['Node'] = None, path_from_parent: Optional[np.ndarray] = None):
        self.state = state
        self.parent = parent
        self.path_from_parent = path_from_parent  # (K, 6)
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
            node = random.choice(nodes)
            hist_arr = np.stack(self._collect_history(node, limit=self.cfg["history_size"]), axis=0)
            actions = self.sample_actions(hist_arr, goal=(gx, gy))
            traj_states, valid = self._simulate_segment(node.state, actions)
            if not valid:
                continue

            new_state = CarState(*traj_states[-1])
            new_node = Node(new_state, parent=node, path_from_parent=traj_states)
            node.children.append(new_node)
            nodes.append(new_node)

            if math.hypot(new_state.x - gx, new_state.y - gy) <= self.cfg["goal_tolerance"]:
                goal_node = new_node
                break

        return nodes, goal_node

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

    @torch.no_grad()
    def sample_actions(self, history_np: np.ndarray, goal: Tuple[float, float]) -> np.ndarray:
        H = self.cfg["history_size"]
        P = self.cfg["pred_horizon"]
        A = self.cfg["action_horizon"]
        D = self.cfg["action_dim"]
        crop = self.cfg["map_crop_size"]
        device = self.device

        last = history_np[-1]
        cx, cy = float(last[0]), float(last[1])
        local = crop_local_map(self.env.grid, cx, cy, crop)
        local_t = torch.from_numpy(local).unsqueeze(0).to(device)
        hist_np = history_np.copy()
        mode = self.cfg.get("history_mode", "full")
        if mode == "last":
            hist_np = np.tile(hist_np[-1:], (hist_np.shape[0], 1))
        elif mode == "none":
            hist_np = np.zeros_like(hist_np)
        hist_t = torch.from_numpy(hist_np).unsqueeze(0).to(device)
        goal_t = torch.tensor([[goal[0], goal[1]]], dtype=torch.float32, device=device)

        seq_dim = P * D
        x_t = torch.randn(1, seq_dim, device=device)

        for t_step in reversed(range(self.sched.T)):
            t = torch.tensor([t_step], device=device, dtype=torch.long)
            cond = dict(history=hist_t, goal=goal_t, local_map=local_t)
            x_t = self.sched.p_sample_step(self.model.forward, x_t, t, cond)

        x0 = x_t.view(1, P, D)
        actions = x0[:, :A, :].clamp_(
            torch.tensor([-self.env.max_speed, -self.env.max_steer], device=device),
            torch.tensor([ self.env.max_speed,  self.env.max_steer], device=device)
        )
        return actions.squeeze(0).cpu().numpy()

    def _simulate_segment(self, start: CarState, actions: np.ndarray) -> Tuple[np.ndarray, bool]:
        states = [start.to_array()]
        cur = start
        for a in actions:
            nxt, collided = self.env.step(cur, (float(a[0]), float(a[1])))
            if collided:
                return np.stack(states, axis=0), False
            states.append(nxt.to_array())
            cur = nxt
        return np.stack(states, axis=0), True


# ----------------------------- Inference / Evaluation -----------------------------

def load_checkpoint(ckpt_path: str, cfg: Dict) -> Tuple[ConditionalDiffusionModel, DiffusionScheduler]:
    ckpt = torch.load(ckpt_path, map_location=cfg["device"])
    saved_cfg = ckpt.get("cfg", cfg)
    merged = {**saved_cfg, **cfg}
    model, sched = build_model_and_sched(merged)
    model.load_state_dict(ckpt["model"])
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


def collect_edges(nodes: List[Node]) -> List[List[Tuple[float, float]]]:
    edges: List[List[Tuple[float, float]]] = []
    for n in nodes:
        if n.parent is None or n.path_from_parent is None:
            continue
        pts = [(float(s[0]), float(s[1])) for s in n.path_from_parent]
        edges.append(pts)
    return edges


def visualize_map_and_path(grid: np.ndarray, edges: List[List[Tuple[float, float]]],
                           best_path: List[Tuple[float, float]], out_png: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid)  # default colormap
    # plot edges
    for e in edges:
        xs = [p[0] for p in e]
        ys = [p[1] for p in e]
        ax.plot(xs, ys, linewidth=0.8)
    # plot best path thicker
    if best_path:
        xs = [p[0] for p in best_path]
        ys = [p[1] for p in best_path]
        ax.plot(xs, ys, linewidth=2.5)
    ax.set_xlim([-0.5, grid.shape[1]-0.5])
    ax.set_ylim([grid.shape[0]-0.5, -0.5])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("DiTree with Best Path")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
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
            start = tuple(map(int, d["start_pos"].tolist()))
            goal = tuple(map(int, d["goal_pos"].tolist()))

        env = GridCarEnv(grid, dt=cfg["dt"], wheelbase=cfg["wheelbase"],
                         max_speed=cfg["max_speed"], max_steer=cfg["max_steer"])
        model, sched = load_checkpoint(ckpt_path, cfg)
        planner = DiffusionTreePlanner(env, model, sched, cfg)

        nodes, goal_node = planner.build_tree(start, goal)
        reached = goal_node is not None
        edges = collect_edges(nodes)
        best_path = extract_best_path(goal_node, nodes, goal_xy=(goal[1], goal[0]))

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

        # save visualization
        if visualize:
            visualize_map_and_path(grid, edges, best_path, os.path.join(out_sub, "snapshot.png"))

        row = {
            "file": os.path.basename(path),
            "nodes_expanded": len(nodes),
            "reached_goal": int(reached),
            "best_path_len": len(best_path),
        }
        all_rows.append(row)
        print(f"[TEST] {row}")

    # summary CSV
    summary_csv = os.path.join(out_dir, "summary.csv")
    import csv
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "nodes_expanded", "reached_goal", "best_path_len"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote summary: {summary_csv}")
    return summary_csv


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=0, help="Number of training epochs (0 = skip training).")
    parser.add_argument("--eval_all_tests", action="store_true", help="Run inference over ALL test NPZ files.")
    parser.add_argument("--visualize", action="store_true", help="Save PNG visualizations per test map.")
    parser.add_argument("--out_dir", type=str, default="outputs_test", help="Output directory for eval results.")
    parser.add_argument("--history_mode", type=str, default="full", choices=["full","last","none"], help="Inference conditioning: full/last/none")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["history_mode"] = args.history_mode
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    print("Using device:", cfg["device"])
    print("Train dir:", cfg["train_data_dir"])
    print("Test dir:", cfg["test_data_dir"])

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
        nodes, goal_node = planner.build_tree(start, goal)
        edges = collect_edges(nodes)
        best_path = extract_best_path(goal_node, nodes, goal_xy=(goal[1], goal[0]))

        demo_dir = os.path.join(args.out_dir, "demo_single")
        os.makedirs(demo_dir, exist_ok=True)
        np.savetxt(os.path.join(demo_dir, "best_path.csv"), np.array(best_path, dtype=np.float32), delimiter=",", header="x,y", comments="")
        with open(os.path.join(demo_dir, "ditree_edges.json"), "w", encoding="utf-8") as f:
            json.dump([{"points": e} for e in edges], f, indent=2)
        if args.visualize:
            visualize_map_and_path(grid, edges, best_path, os.path.join(demo_dir, "snapshot.png"))
        print(f"Single test demo saved under: {demo_dir}")


if __name__ == "__main__":
    main()
