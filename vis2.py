import open3d as o3d
import numpy as np
#import rerun as rr
import trimesh

original_cloud_path = "./output/Pheno4D/Shapenet_selected_inner_points1.obj"

processed_cloud_path = "./output/Pheno4D/Shapenet_processed1.obj"

original_cloud = trimesh.load(original_cloud_path)
processed_cloud = trimesh.load(processed_cloud_path)

pc_original = o3d.geometry.PointCloud()
pc_original.points = o3d.utility.Vector3dVector(original_cloud.vertices)
pc_original.colors = o3d.utility.Vector3dVector(np.ones((len(original_cloud.vertices), 3)) * [0, 1, 0])

pc_processed = o3d.geometry.PointCloud()
pc_processed.points = o3d.utility.Vector3dVector(processed_cloud.vertices)
pc_processed.colors = o3d.utility.Vector3dVector(np.ones((len(processed_cloud.vertices), 3)) * [1, 0, 0])

o3d.visualization.draw_geometries([pc_original, pc_processed])

# o3d.visualization.draw_geometries([original_cloud, processed_cloud])

# rr.init("my_app", spawn=True)

# rr.log("point_cloud", rr.Points3D(point_set))