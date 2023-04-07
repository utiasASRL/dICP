""" 
    Various ICP algorithms for point clouds.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from diff_ICP.kd_nn import KDTree
from diff_ICP.diff_nn import diff_nn

def ptp_ICP(source_tree, target_tree, max_iterations=100, tolerance=1e-12):
    """
    Point-to-point ICP algorithm.
    :param source_tree: Source point cloud tree resolved in source frame ps_s.
    :param target_tree: Target point cloud tree resolved in target frame pt_t.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Tolerance for convergence.
    :return: Transformed source point cloud and transformation from source to target T_ts.
    """
    # Check if source and target are KDTree objects
    if not isinstance(source_tree, KDTree):
        # Convert to KDTree
        source_tree = KDTree(source_tree.tolist())
    if not isinstance(target_tree, KDTree):
        # Convert to KDTree
        target_tree = KDTree(target_tree.tolist())

    # Initialize transformation matrix
    T_ts = np.identity(4)
    # Source points are resolved in s frame
    ps_s = np.array(source_tree.points)

    # Iterate until convergence
    for ii in range(max_iterations):
        # Find nearest neighbour for each point in source, these points are in target frame
        nn_t = np.array([target_tree.nearest_neighbour(point) for point in ps_s])
        
        # Compute centroids
        mean_ps_s = np.mean(ps_s, axis=0).reshape(3, 1)
        mean_pt_t = np.mean(nn_t, axis=0).reshape(3, 1)

        # Compute covariance matrix
        W_st = (nn_t.T - mean_pt_t) @ (ps_s.T - mean_ps_s).T
        W_st = W_st / len(source_tree.points)

        # Compute SVD
        U, D, Vt = np.linalg.svd(W_st)

        # Compute rotation matrix from source to "target frame"
        # This frame is not the true target frame, but we are trying to get there
        C_ts = U @ np.diag([1, 1, np.linalg.det(U)*np.linalg.det(Vt.T)]) @ Vt

        # Compute translation
        r_st_t = mean_pt_t - C_ts @ mean_ps_s

        # Update transformation matrix, remember frame t here is the new predicted target frame
        # T_ts_update has frame s, which is the old best target frame, and frame t, which is the new predicted target frame
        T_ts_update = np.vstack((np.hstack((C_ts, r_st_t)), np.array([0, 0, 0, 1])))
        T_ts = T_ts_update @ T_ts

        # Update source point cloud
        ps_s = (C_ts @ ps_s.T + r_st_t).T

        # Check convergence, if converged then the updated s frame is aligned with the true t frame
        # This means that T_ts is now composed of the transformation from the original s to the true t
        if np.sum((ps_s - nn_t) ** 2) < tolerance:
            break

    return KDTree(ps_s.tolist()), T_ts

def diff_ptp_ICP(source, target, max_iterations=100, tolerance=1e-8):
    """
    Differentiable point-to-point ICP algorithm.
    :param source: A pytorch tensor of shape (n, d) representing the source point cloud.
    :param target: A pytorch tensor of shape (n, d) representing the target point cloud.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Tolerance for convergence.
    :return: Transformed source point cloud and transformation from source to target T_ts.
    :return: Transformation from source to target T_ts.
    """
    # Initialize transformation matrix
    T_ts = torch.eye(4, dtype=torch.float64)
    # Source points are resolved in s frame
    ps_s = source

    # Iterate until convergence
    for ii in range(max_iterations):
        # Find nearest neighbour for each point in source, these points are in target frame
        nn_t = torch.zeros_like(source)
        for jj in range(len(source)):
            nn_t[jj] = diff_nn(ps_s[jj].reshape(1, 3), target)

        # Compute centroids
        mean_ps_s = torch.mean(ps_s, dim=0).reshape(3, 1)
        mean_pt_t = torch.mean(nn_t, dim=0).reshape(3, 1)

        # Compute covariance matrix
        W_st = (nn_t.T - mean_pt_t) @ (ps_s.T - mean_ps_s).T
        W_st = W_st / len(source)

        # Compute SVD
        U, D, Vt = torch.svd(W_st)

        # Compute rotation matrix from source to "target frame"
        # This frame is not the true target frame, but we are trying to get there
        C_ts = U @ torch.diag(torch.tensor([1, 1, torch.det(U) * torch.det(Vt.T)])) @ Vt

        # Compute translation
        r_st_t = mean_pt_t - C_ts @ mean_ps_s

        # Update transformation matrix, remember frame t here is the new predicted target frame
        # T_ts_update has frame s, which is the old best target frame, and frame t, which is the new predicted target frame
        T_ts_update = torch.vstack((torch.hstack((C_ts, r_st_t)), torch.tensor([0, 0, 0, 1])))
        T_ts = T_ts_update @ T_ts

        # Update source point cloud
        ps_s = (C_ts @ ps_s.T + r_st_t).T

        # Check convergence, if converged then the updated s frame is aligned with the true t frame
        # This means that T_ts is now composed of the transformation from the original s to the true t
        if torch.sum((ps_s - nn_t) ** 2) < tolerance:
            break

    return ps_s, T_ts

def plot_overlay(points1, points2, map=None):
    # Check if points are torch tensors, if so then convert to numpy
    if isinstance(points1, torch.Tensor):
        points1 = points1.detach().numpy()
    if isinstance(points2, torch.Tensor):
        points2 = points2.detach().numpy()
    
    # Plot points
    x_self = [point[0] for point in points1]
    y_self = [point[1] for point in points1]
    plt.scatter(x_self, y_self, marker='o', color='b')

    x_other = [point[0] for point in points2]
    y_other = [point[1] for point in points2]
    plt.scatter(x_other, y_other, marker='o', color='r')

    # Find bounds for  plotting
    if map is not None:
        xlim, ylim = map.get_boundingbox()
        x_min = xlim[0]
        x_max = xlim[1]
        y_min = ylim[0]
        y_max = ylim[1]
    else:
        max_val = max(max(x_self), max(x_other), max(y_self), max(y_other))
        min_val = min(min(x_self), min(x_other), min(y_self), min(y_other))
        x_min = min_val - 2
        x_max = max_val + 2
        y_min = min_val - 2
        y_max = max_val + 2

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def plot_map(points, color='b', map=None):
    # Check if points are torch tensors, if so then convert to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().numpy()
    
    # Plot points
    x_self = [point[0] for point in points]
    y_self = [point[1] for point in points]
    plt.scatter(x_self, y_self, marker='o', color=color)

    # Find bounds for  plotting
    if map is not None:
        xlim, ylim = map.get_boundingbox()
        x_min = xlim[0]
        x_max = xlim[1]
        y_min = ylim[0]
        y_max = ylim[1]
    else:
        max_val = max(max(x_self), max(y_self))
        min_val = min(min(x_self), min(y_self))
        x_min = min_val - 2
        x_max = max_val + 2
        y_min = min_val - 2
        y_max = max_val + 2

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()