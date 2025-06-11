# Author: Frank ZY Dou
import os
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from utils import  save_obj,read_VD, read_point, winding_number
from scipy.optimize import milp, Bounds, LinearConstraint
import open3d as o3d

def sample_points(point_set, num_points):
    return point_set[np.random.choice(np.arange(len(point_set)), num_points, replace=False)]



def remove_outliers(point_set):
    # remove z point less than 0.01
    point_set = point_set[point_set[:, 2] > 0.1]
    return point_set



def normalize_point_set(point_set):
    # normalize the point set between 0 and 1   
    point_set = point_set - np.min(point_set, axis=0)
    point_set = point_set / np.max(np.linalg.norm(point_set, axis=1))
    # point_set = point_set - np.mean(point_set, axis=0)
    # rotate the point set to the z-axis
    # point_set = rotate_point_set(point_set)
    return point_set

def rotate_point_set(point_set):
    # rotate the point set to the z-axis
    point_set = point_set - np.mean(point_set, axis=0)
    point_set = point_set / np.max(np.linalg.norm(point_set, axis=1))
    return point_set



if __name__ == "__main__":

    real_name = '01Ants-12_pc'
    dilation = 0.025
    inner_points = "random"
    max_time_SCP = 100 # in second
    output_path = './output/ScanNet'
    # point_set = trimesh.load('./input/%s.obj'%real_name)
    path = './processed_dataset/ScanNet/0059_00.pcd'
    # point_set = np.load(path)
    point_set = o3d.io.read_point_cloud(path)
    # point_set = np.load(path)
    # point_set = np.array(point_set.points)
    # point_set = remove_outliers(point_set)
    # point_set = sample_points(point_set, 3000)
    point_set = normalize_point_set(point_set.points)

# Create an Open3D point cloud object
    vis_point_set = o3d.geometry.PointCloud()
    vis_point_set.points = o3d.utility.Vector3dVector(point_set)


    # visualize the new point set
    o3d.visualization.draw_geometries([vis_point_set])

    # import pdb; pdb.set_trace()


    inner_points = point_set[np.random.choice(np.arange(len(point_set)), 30000)]
    # inner_points = normalize_point_set(inner_points)

    point_set = point_set[np.random.choice(np.arange(len(point_set)), 3000)]
    # point_set = normalize_point_set(point_set)
    
    # save processed point cloud
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_obj(os.path.join(output_path, "ScanNet_processed2.obj"), inner_points)
    # save_obj(os.path.join(output_path, "Shapenet_processed2.obj"), point_set)e

    # i/nner_points = inner_points[np.random.choice(np.arange(len(inner_points)), 30000)] # downsample inner points to 50000.
    print("The number of sampled inner candidates: ", len(inner_points))
    print("The number of surface samples: ", len(point_set))
    inner_points_g = torch.tensor(inner_points).cuda().double()
    point_set_g = torch.tensor(point_set).cuda().double()
    dist = torch.cdist(inner_points_g, point_set_g, p=2)
    radius = dist.topk(1, largest=False).values
    radius = radius + dilation

    # import pdb; pdb.set_trace()
    # save_obj("./output/pc_samples.obj", point_set) # to be covered surface samples.
    # save_obj("./output/pc_inner_points.obj", inner_points) # candidate inner points.

    # Coverage Matrix -> GPU.
    point_set_g = torch.tensor(point_set).cuda().double()
    innerpoints_g = torch.tensor(inner_points).cuda().double()
    radius_g = torch.tensor(radius).cuda().double()
    radius_g = radius_g[:,0]
    radius_g = radius_g.unsqueeze(0).repeat(len(point_set), 1)
    D = torch.cdist(point_set_g, innerpoints_g, p=2)
    D = torch.gt(radius_g, D).type(torch.int)
    D = D.cpu().numpy()
    # Done

    c = np.ones(len(inner_points))
    options = {"disp": True, "time_limit": max_time_SCP, }
    A,b =  D, np.ones(len(point_set))
    integrality = np.ones(len(inner_points))
    lb, ub = np.zeros(len(inner_points)), np.ones(len(inner_points))
    variable_bounds = Bounds(lb, ub)
    constraints = LinearConstraint(A, lb=b)
    res_milp = milp(
        c,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=constraints,
        options=options)

    res_milp.x = [int(x_i) for x_i in res_milp.x]
    print(res_milp)
    print(np.sum(res_milp.x))
    value_pos = np.nonzero(res_milp.x)[0]
    print("The number of selected inner points: ", len(value_pos))





    # visualize the selected inner points
    vis_inner_points = o3d.geometry.PointCloud()
    vis_inner_points.points = o3d.utility.Vector3dVector(inner_points[value_pos])
    o3d.visualization.draw_geometries([vis_inner_points])


    # import pdb; pdb.set_trace()
    save_obj(os.path.join(output_path, "ScanNet_selected_inner_points2.obj"), inner_points[value_pos])




