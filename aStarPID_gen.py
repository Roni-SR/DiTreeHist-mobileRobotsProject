import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import math
import heapq
import os
import glob
import logging
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

# --- General Settings ---
MAZES_SOURCE_DIR = "generated_mazes"  # Path to the source maps in CSV format
OUTPUT_DIR = "NewPID_maps"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_maps")
TEST_DIR = os.path.join(OUTPUT_DIR, "test_maps")
LOG_FILE = os.path.join(OUTPUT_DIR, "map_generation.log")
NUM_SUBOPTIMAL_PATHS = 3  # Number of suboptimal paths to generate for each map
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% for training, 20% for testing

# Configure Logger to save the execution process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)


# --- 1. A* Algorithm for Optimal Path Finding ---

class AStarNode:
    """A node for the A* algorithm with cost and heuristic."""

    def __init__(self, pos: Tuple[int, int], parent=None, g_cost=0, h_cost=0):
        self.pos = pos
        self.parent = parent
        self.g = g_cost  # Cost from start
        self.h = h_cost  # Heuristic to goal
        self.f = self.g + self.h  # Total cost

    def __lt__(self, other):
        """Comparison for the min-heap."""
        return self.f < other.f


def a_star_path_planner(grid: np.ndarray, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> Optional[
    List[Tuple[int, int]]]:
    """Plans an optimal path using the A* algorithm."""
    rows, cols = grid.shape

    def heuristic(pos: Tuple[int, int]) -> float:
        return math.hypot(pos[0] - goal_pos[0], pos[1] - goal_pos[1])

    start_node = AStarNode(start_pos, h_cost=heuristic(start_pos))
    open_set = [start_node]
    closed_set = set()
    g_scores = {start_pos: 0}

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.pos == goal_pos:
            path = []
            while current_node:
                path.append(current_node.pos)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.pos)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor_pos = (current_node.pos[0] + dx, current_node.pos[1] + dy)

            if not (0 <= neighbor_pos[0] < rows and 0 <= neighbor_pos[1] < cols) or \
                    grid[neighbor_pos[0], neighbor_pos[1]] == 1 or \
                    neighbor_pos in closed_set:
                continue

            tentative_g_score = g_scores[current_node.pos] + math.hypot(dx, dy)

            if tentative_g_score < g_scores.get(neighbor_pos, float('inf')):
                g_scores[neighbor_pos] = tentative_g_score
                h_score = heuristic(neighbor_pos)
                neighbor_node = AStarNode(neighbor_pos, parent=current_node, g_cost=tentative_g_score, h_cost=h_score)
                heapq.heappush(open_set, neighbor_node)
    return None


# --- 2. Vehicle Simulation and PID Controller ---

def ackermann_model(x, y, theta, v, delta, dt, L=2.0):
    """Simulates a step of the Ackermann vehicle model."""
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = theta + (v / L) * math.tan(delta) * dt
    return x_new, y_new, theta_new


def track_path_pid(grid: np.ndarray, start_pos: Tuple[int, int], path: List[Tuple[int, int]], speed=1.0) -> Optional[
    np.ndarray]:
    """Simulates path tracking using a simple PID controller."""
    if not path or len(path) < 2: return None

    # Convert grid coordinates to continuous coordinates
    current_x, current_y = float(start_pos[1]), float(start_pos[0])

    # Initial heading
    target_x, target_y = path[1]
    current_theta = math.atan2(target_y - current_y, target_x - current_x)

    # Controller settings
    kp, ki, kd, dt = 1.2, 0.05, 0.2, 0.1
    integral, prev_error = 0, 0

    followed_path = [[current_x, current_y, current_theta, speed, 0, 0]]  # x, y, yaw, v, throttle, steer
    path_index = 1

    for _ in range(len(path) * 20):  # Iteration limit
        if path_index >= len(path): break

        target_x, target_y = path[path_index]
        target_heading = math.atan2(target_y - current_y, target_x - current_x)

        error = target_heading - current_theta
        error = (error + math.pi) % (2 * math.pi) - math.pi  # Normalize angle

        integral += error * dt
        derivative = (error - prev_error) / dt
        steer_angle = kp * error + ki * integral + kd * derivative
        prev_error = error

        current_x, current_y, current_theta = ackermann_model(current_x, current_y, current_theta, speed, steer_angle,
                                                              dt)

        # Collision check
        if grid[int(round(current_y)), int(round(current_x))] == 1:
            logging.warning("PID controller resulted in a collision. Path truncated.")
            break

        followed_path.append([current_x, current_y, current_theta, speed, 0, steer_angle])

        if math.hypot(current_x - target_x, current_y - target_y) < 1.0:
            path_index += 1

    return np.array(followed_path)


# --- 3. Generation of Suboptimal Paths ---

def generate_suboptimal_path(grid: np.ndarray, optimal_path: List[Tuple[int, int]], goal_pos: Tuple[int, int]) -> \
Optional[List[Tuple[int, int]]]:
    """Generates a suboptimal path by a random deviation and re-planning."""
    if len(optimal_path) < 10: return None

    rows, cols = grid.shape
    random_move_index = random.randint(3, len(optimal_path) - 4)
    original_pos = optimal_path[random_move_index]

    for _ in range(10):  # Attempts to find a valid point
        dx, dy = random.randint(-5, 5), random.randint(-5, 5)
        new_pos = (original_pos[0] + dx, original_pos[1] + dy)
        if 0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols and grid[new_pos[0], new_pos[1]] == 0:
            re_planned_path = a_star_path_planner(grid, new_pos, goal_pos)
            if re_planned_path:
                return optimal_path[:random_move_index] + re_planned_path
    return None


# --- 4. Helper Functions ---

def get_random_empty_point(grid: np.ndarray) -> Tuple[int, int]:
    """Finds a random empty point on the map."""
    rows, cols = grid.shape
    while True:
        y = random.randint(1, rows - 2)
        x = random.randint(1, cols - 2)
        if grid[y, x] == 0:
            return (y, x)


def plot_and_save_scenario(grid, start_pos, goal_pos, paths_dict, file_name):
    """Creates and saves an image of the map and paths."""
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=ListedColormap(['white', 'black']), origin='lower')
    plt.plot(start_pos[1], start_pos[0], 'go', markersize=10, label='Start')
    plt.plot(goal_pos[1], goal_pos[0], 'rx', markersize=10, label='Goal')

    colors = ['orange', 'blue', 'green', 'cyan', 'magenta', 'purple']
    for i, (name, path) in enumerate(paths_dict.items()):
        if path is not None:
            plt.plot(path[:, 1], path[:, 0], color=colors[i % len(colors)], label=name)

    plt.legend()
    plt.grid(False)
    plt.savefig(file_name)
    plt.close()


# --- 5. Main Data Generation Function ---

def generate_map_data(map_file_path: str, output_path: str):
    """
    Loads a map, generates various paths, and saves everything to an NPZ file.
    """
    try:
        grid = np.loadtxt(map_file_path, delimiter=',', dtype=int)
    except Exception as e:
        logging.error(f"Failed to load map {map_file_path}. Error: {e}")
        return False

    for _ in range(10):  # 10 attempts to find valid start/goal points
        start_pos = get_random_empty_point(grid)
        goal_pos = get_random_empty_point(grid)

        optimal_path_coords = a_star_path_planner(grid, start_pos, goal_pos)

        if optimal_path_coords and len(optimal_path_coords) > 10:
            paths_to_save = {}

            # 1. Optimal Path (A*)
            pid_optimal = track_path_pid(grid, start_pos, optimal_path_coords)
            if pid_optimal is not None:
                paths_to_save['optimal_path'] = pid_optimal

            # 2. Suboptimal Paths
            for i in range(NUM_SUBOPTIMAL_PATHS):
                sub_path_coords = generate_suboptimal_path(grid, optimal_path_coords, goal_pos)
                if sub_path_coords:
                    pid_suboptimal = track_path_pid(grid, start_pos, sub_path_coords)
                    if pid_suboptimal is not None:
                        paths_to_save[f'suboptimal_path_{i + 1}'] = pid_suboptimal

            if 'optimal_path' in paths_to_save:
                # Save the data to an NPZ file
                np.savez_compressed(
                    output_path,
                    grid=grid,
                    start_pos=np.array(start_pos),
                    goal_pos=np.array(goal_pos),
                    **paths_to_save
                )
                logging.info(f"Successfully generated and saved data to {output_path}")
                return True

    logging.warning(f"Failed to generate valid paths for map {map_file_path} after multiple attempts.")
    return False


if __name__ == "__main__":
    # Create output directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    maze_files = glob.glob(os.path.join(MAZES_SOURCE_DIR, "*.csv"))
    if not maze_files:
        logging.error(f"No CSV maze files found in {MAZES_SOURCE_DIR}")
    else:
        random.shuffle(maze_files)
        split_index = int(len(maze_files) * TRAIN_TEST_SPLIT_RATIO)
        train_files = maze_files[:split_index]
        test_files = maze_files[split_index:]

        logging.info(
            f"Found {len(maze_files)} maps. Splitting into {len(train_files)} train and {len(test_files)} test maps.")

        # Generate training maps
        logging.info("\n--- Generating Training Maps ---")
        for i, f in enumerate(tqdm(train_files, desc="Generating Train NPZ")):
            output_path = os.path.join(TRAIN_DIR, f"map_data_{i + 1}.npz")
            generate_map_data(f, output_path)

        # Generate test maps
        logging.info("\n--- Generating Test Maps ---")
        for i, f in enumerate(tqdm(test_files, desc="Generating Test NPZ")):
            output_path = os.path.join(TEST_DIR, f"map_data_{i + 1}.npz")
            generate_map_data(f, output_path)

        logging.info("\nMap generation process finished.")
