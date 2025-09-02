import numpy as np
import os
import glob

#d = np.load("NewPID_maps/test_maps/map_data_1.npz", allow_pickle=True)
#print(d.files)
#print(d["grid"].shape)
#print(d["grid"].astype(np.uint8))

#test_files = sorted(glob.glob(os.path.join("demo_single", "*.npz")))
test_files = [f for f in glob.glob("demo_single/*.npz") if "grid" in np.load(f, allow_pickle=True).files]
#test_files = sorted("NewPID_maps/test_maps", "*.npz")

# load model + planner (reinitialized per grid size)
for path in test_files:
    with np.load(path, allow_pickle=True) as d:
        print("evaluating:", path)
        grid = d["grid"].astype(np.uint8)
        start = tuple(map(int, d["start_pos"].tolist()))
        goal = tuple(map(int, d["goal_pos"].tolist()))
        print("Grid:\n", grid)
        print("Start:", start)
        print("Goal:", goal)
