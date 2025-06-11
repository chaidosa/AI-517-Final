#!/usr/bin/env python3
# save_visualisations.py  —  v4
# ------------------------------------------------------------
# • “SAT” folders   : render *candidate_inner_points* (green) vs *selected_inner_points* (red)
# • non-SAT folders : render *processed*              (green) vs *selected_inner_points* (red)
# • PNGs saved as: <SOURCE_FOLDER>_<BASE_ID>_<SAT|MLP>.png
# ------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageChops

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE_DIR    = Path("./output")
FOLDERS     = [
    "ScanNet", "ScanNet_SAT",
    "ShapeNet", "ShapeNet_SAT",
    "Pheno4D",  "Pheno4D_SAT",
]
RESULTS_DIR = Path("./final-figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Auto-crop white margins (Pillow)
# ------------------------------------------------------------------
def autocrop_white(fn: Path):
    img = Image.open(fn)
    # white background
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        img.crop(bbox).save(fn)

# ------------------------------------------------------------------
# Helper – Trimesh → coloured Open3D point cloud
# ------------------------------------------------------------------
def mesh_to_pcd(mesh: trimesh.Trimesh, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(rgb, (len(mesh.vertices), 1))
    )
    return pcd

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
for folder in FOLDERS:
    folder_path = BASE_DIR / folder
    if not folder_path.is_dir():
        print(f"Missing folder: {folder_path} – skipped.")
        continue

    is_sat = "SAT" in folder.upper()

    if is_sat:
        search_glob  = "*candidate_inner_points*.obj"
        partner_from = "candidate_inner_points"
        partner_to   = "selected_inner_points"
        green_label  = "candidate"
    else:
        search_glob  = "*processed*.obj"
        partner_from = "processed"
        partner_to   = "selected_inner_points"
        green_label  = "processed"

    for green_path in folder_path.glob(search_glob):
        base_id  = green_path.stem.replace(partner_from, "")
        red_name = green_path.name.replace(partner_from, partner_to)
        red_path = folder_path / red_name

        if not red_path.exists():
            print(f"No partner for {green_path.name} → expected {red_name}")
            continue

        # ---- load meshes ----
        try:
            mesh_green = trimesh.load(green_path, process=False)
            mesh_red   = trimesh.load(red_path,   process=False)
        except Exception as e:
            print(f" Failed loading {green_path.name}: {e}")
            continue

        pcd_green = mesh_to_pcd(mesh_green, rgb=[0,1,0])  # green
        pcd_red   = mesh_to_pcd(mesh_red,   rgb=[1,0,0])  # red

        # ---- off-screen render ----
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=base_id,
            width=800, height=800,
            visible=False
        )
        vis.get_render_option().background_color = np.array([1,1,1])
        vis.add_geometry(pcd_green)
        vis.add_geometry(pcd_red)
        vis.poll_events()
        vis.update_renderer()

        suffix   = "SAT" if is_sat else "MLP"
        # <-- include the folder name in the PNG filename here -->
        out_file = RESULTS_DIR / f"{folder}_{base_id}_{suffix}.png"

        vis.capture_screen_image(str(out_file), do_render=True)
        vis.destroy_window()

        # autocrop_white(out_file)

        print(f"{green_label:9s} vs selected → {out_file}")

print("Done")
