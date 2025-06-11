# Author: Frank ZY Dou
# SAT method: Binary Search (from user contribution)
# Combined and refined by Gemini
import os
import time
import torch
import trimesh
import numpy as np
from tqdm import tqdm
import open3d as o3d

# PySAT imports for the binary search method
from pysat.formula import CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Minisat22

# --- Configuration (from user's script) ---
DILATION = 0.025
INNER_SUBSAMP = 30000
SURFACE_SUBSAMP = 5000
EARLY_STOP_K = 950 # Stop search if a solution with k < this value is found
TIMEOUT_MINUTES = 8 # Stop search after this many minutes

def sample_points(point_set, num_points):
    return point_set[np.random.choice(np.arange(len(point_set)), num_points, replace=False)]


def remove_outliers(point_set):
    # remove z point less than 0.01
    point_set = point_set[point_set[:, 2] > 0.1]
    return point_set




def save_obj(filename, points):
    """Saves a point cloud to an OBJ file."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

def normalize_point_set(point_set):
    """Normalizes the point set to fit within a unit sphere centered at the origin."""
    if point_set.shape[0] == 0:
        return point_set
    point_set = point_set - np.min(point_set, axis=0)
    max_norm = np.max(np.linalg.norm(point_set, axis=1))
    if max_norm > 1e-6:
        point_set = point_set / max_norm
    return point_set

if __name__ == "__main__":

    # --- 1. Load and Process Data ---
    output_path = './output/ScanNet_SAT'
    data_path = './processed_dataset/ScanNet/0059_00.pcd'
    
    print("Loading point cloud...")
    try:
        # point_set = np.load(data_path)
        point_set = o3d.io.read_point_cloud(data_path)
        point_set = np.array(point_set.points)
        # point_set = remove_outliers(point_set)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Creating a dummy sphere point cloud to demonstrate the script.")
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        point_set = mesh.vertices
    # point_set = sample_points(point_set, 3000)
    point_set = normalize_point_set(point_set)

    if len(point_set) > 0:
        num_inner = min(len(point_set), INNER_SUBSAMP)
        num_surface = min(len(point_set), SURFACE_SUBSAMP)
        inner_points = point_set[np.random.choice(np.arange(len(point_set)), num_inner, replace=False)]
        point_set_surface = point_set[np.random.choice(np.arange(len(point_set)), num_surface, replace=False)]
    else:
        print("Error: Point cloud is empty after processing.")
        exit()

    print(f"# Surface samples   : {len(point_set_surface):,}")
    print(f"# Candidate inners  : {len(inner_points):,}")

    # --- 2. Build Coverage Matrix (D) and Radii ---
    print("Building coverage matrix on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    s_gpu = torch.from_numpy(point_set_surface).to(device).double()
    i_gpu = torch.from_numpy(inner_points).to(device).double()
    
    # Compute radii for each inner point
    nn_dist = torch.cdist(i_gpu, s_gpu, p=2).min(dim=1).values
    radii = (nn_dist + DILATION).cpu().numpy()

    # Build the coverage matrix D
    dist_gpu = torch.cdist(s_gpu, i_gpu, p=2)
    R_gpu = torch.from_numpy(radii).to(device).double().unsqueeze(0)
    D = (dist_gpu <= R_gpu).cpu().numpy().astype(np.int8)
    del s_gpu, i_gpu, dist_gpu, R_gpu, nn_dist # Free GPU memory

    Ns, Ni = D.shape
    print(f"Coverage matrix      : {Ns:,} × {Ni:,}")

    # --- 3. Encode the Base CNF (one clause per surface sample) ---
    cnf = CNF()
    for s in tqdm(range(Ns), desc="Building Base CNF"):
        clause = [j + 1 for j in range(Ni) if D[s, j]]
        if not clause:
            print(f"Warning: Surface point {s} is not covered by any candidate inner point. The problem may be UNSAT.")
            continue
        cnf.append(clause)
    
    print(f"Base CNF             : {Ni:,} vars, {len(cnf.clauses):,} clauses")

    # --- 4. Binary Search on Cardinality (k) for the Optimal Cover ---
    print(f"\nStarting binary search for minimum cardinality cover (Timeout: {TIMEOUT_MINUTES} minutes)...")
    lo, hi = 1, Ni
    best_model = None
    best_k = Ni
    search_start_time = time.monotonic()
    timeout_seconds = TIMEOUT_MINUTES * 60
    reason_for_stop = "found optimal solution"

    while lo <= hi:
        # Check for timeout at the beginning of each iteration
        elapsed_time = time.monotonic() - search_start_time
        if elapsed_time > timeout_seconds:
            reason_for_stop = f"exceeded time limit of {TIMEOUT_MINUTES} minutes"
            print(f"\nTIMEOUT: Search {reason_for_stop}. Using best solution found so far (k={best_k}).")
            break

        k = (lo + hi) // 2
        card = CardEnc.atmost(lits=list(range(1, Ni + 1)), bound=k, encoding=EncType.cardnetwrk)
        
        with Minisat22(bootstrap_with=cnf.clauses + card.clauses) as solver:
            is_sat = solver.solve()
            status = "SAT" if is_sat else "UNSAT"
            print(f"  k = {k:<5}  →  {status}  (time: {elapsed_time:.1f}s)")

            if is_sat:
                best_model = solver.get_model()
                best_k = k
                hi = k - 1
                
                if k < EARLY_STOP_K:
                    reason_for_stop = f"found solution with k={k} < early stop threshold {EARLY_STOP_K}"
                    print(f"EARLY STOP: {reason_for_stop.capitalize()}.")
                    break
            else:
                lo = k + 1
    
    # --- 5. Process and Save Results ---
    if best_model is None:
        print("\nERROR: No solution found. Try increasing DILATION or using more sample points.")
    else:
        # **FIX**: Filter the model to only include original variables (1 to Ni).
        # The cardinality encoder adds auxiliary variables with higher indices,
        # which must be ignored.
        chosen_indices = [var - 1 for var in best_model if 0 < var <= Ni]
        selected_points = inner_points[chosen_indices]

        print(f"\n--- Search Complete ---")
        print(f"Reason for stopping: {reason_for_stop}.")
        print(f"Best cover size found (k): {best_k}")
        print(f"Selected inner points: {len(selected_points)}")

        # Save results
        save_obj(os.path.join(output_path, "processed_surface_points2.obj"), point_set_surface)
        save_obj(os.path.join(output_path, "candidate_inner_points2.obj"), inner_points)
        save_obj(os.path.join(output_path, "selected_inner_points2.obj"), selected_points)
        print(f"OBJ files saved in {output_path}/")
        
        # Visualize
        print("Visualizing the results... (Close the window to exit)")
        pcd_selected = o3d.geometry.PointCloud()
        pcd_selected.points = o3d.utility.Vector3dVector(selected_points)
        pcd_selected.paint_uniform_color([1.0, 0, 0])

        pcd_surface = o3d.geometry.PointCloud()
        pcd_surface.points = o3d.utility.Vector3dVector(point_set_surface)
        pcd_surface.paint_uniform_color([0.5, 0.5, 0.5])

        o3d.visualization.draw_geometries([pcd_surface, pcd_selected], window_name="Binary Search SAT Result")
