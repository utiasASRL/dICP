""" 
    Various ICP algorithms for point clouds.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from dICP.kd_nn import KDTree
from dICP.diff_nn import diff_nn, diff_nn_single
import torch.nn.functional as F

def pt2pt_ICP(source_tree, target_tree, T_init, max_iterations=100, tolerance=1e-12):
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
    T_ts = T_init
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

    print("ii: ", ii)

    return KDTree(ps_s.tolist()), T_ts

def pt2pt_dICP(source, target, T_init, max_iterations=100, tolerance=1e-8, trim_dist=0.0):
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
    T_ts = T_init
    # Source points are resolved in s frame
    if source.shape[1] == 6:
        ps_s = source[:, 0:3]
    else:
        ps_s = source
    if target.shape[1] == 6:
        target = target[:, 0:3]

    # Iterate until convergence
    for ii in range(max_iterations):
        # Find nearest neighbour for each point in source, these points are in target frame
        nn_t = torch.zeros((source.shape[0], target.shape[1]), dtype=source.dtype)
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


def pt2pl_dICP(source, target, T_init, max_iterations=100, tolerance=1e-8, trim_dist=5.0, huber_delta=1.0, dim=3):
    """
    Point-to-plane ICP algorithm.
    :param source_tree: Source point cloud tree resolved in source frame ps_s [n x 6].
    :param target_tree: Target point cloud tree resolved in target frame pt_t [m x 6].
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Tolerance for convergence.
    :return: Transformed source point cloud and transformation from source to target T_ts.
    """

    # Confirm that source, target, and T_init types match
    assert source.dtype == target.dtype == T_init.dtype

    device = source.device

    # Initialize transformation matrix
    T_ts = T_init
    # Source points are resolved in s frame, ignore normals for source
    if source.shape[1] == 6:
        ps_s = source[:, 0:3]
    else:
        ps_s = source

    # Iterate until convergence
    for ii in range(max_iterations):
        # Find nearest neighbour for each point in source, these points are in target frame
        err = torch.zeros(len(source), 1, dtype=source.dtype, device=device)
        w = torch.zeros(len(source), 1, dtype=source.dtype, device=device)
        J = torch.zeros(len(source), 6, dtype=source.dtype, device=device)
        # Extract T_ts components
        C_ts = T_ts[0:3, 0:3]
        r_st_t = T_ts[0:3, 3:]
        ps_s = ps_s.reshape(-1, 3, 1)

        # Transform points to best guess target frame
        ps_t = C_ts @ ps_s + r_st_t.reshape(1, 3, 1)

        # Find nearest neighbours
        nn = diff_nn(ps_t.reshape(-1, 3), target).squeeze()
        nn_t = nn[:, :3].reshape(-1, 3, 1)
        nn_norm = nn[:, 3:].reshape(-1, 3, 1)

        # Compute errors
        nn_err = ps_t - nn_t
        err = torch.sum(nn_err * nn_norm, axis=(1, 2))

        # Compute weights based on trim distance and huber delta
        steep_fact = 5.0
        if huber_delta not in [0.0, None]:
            #w_huber = torch.where(torch.abs(err) > huber_delta, huber_delta / torch.abs(err), torch.ones_like(err))
            #w_huber = 0.5 * (torch.tanh(steep_fact*(huber_delta - torch.abs(err)) - 3.0) + 1.0) * (1.0 - hub_fact) + hub_fact
            # Use pseudo huber loss
            w_huber = huber_delta**2 / (huber_delta**2 + err**2)
        else:
            w_huber = torch.ones_like(err)

        if trim_dist not in [0.0, None]:
            w_trim = 0.5 * torch.tanh(steep_fact * (trim_dist - torch.linalg.norm(nn_err, axis=1).squeeze()) - 3.0) + 0.5
        else:
            w_trim = torch.ones_like(err)
        w = w_huber * w_trim

        # Compute Jacobian components of err with respect to T_ts
        J_C = torch.bmm(nn_norm.transpose(1,2), skew_operator(C_ts @ ps_s)).transpose(1,2)
        J_r = - nn_norm

        # Reshape err, w, J_C, and J_r to be N-dimensional arrays
        err = err.reshape(-1)
        J_C = J_C.reshape(-1, 3)
        J_r = J_r.reshape(-1, 3)

        # Combine Jacobian components into a single Jacobian matrix
        J = torch.hstack((J_C, J_r))

        # Assemble weight matrix
        W = torch.diag(w.reshape(-1))

        if dim == 2:
            D = torch.zeros((6,3), dtype=source.dtype, device=device)
            D[2,0] = D[3,1] = D[4,2] = 1.0
            J = J @ D

        A = J.T @ W @ J + 1e-12 * torch.eye(J.shape[1], dtype=source.dtype, device=device)

        # Compute update
        del_T_ts = - torch.linalg.inv(A) @ J.T @ W @ err
        if dim == 2:
            temp_step = torch.zeros((6), dtype=source.dtype, device=device)
            temp_step[2:5] = del_T_ts
            del_T_ts = temp_step

        del_C = torch.matrix_exp(skew_operator(del_T_ts[0:3]).squeeze())
        del_r = del_T_ts[3:6].reshape(3, 1)

        # Update T_ts
        T_ts_new = torch.eye(4, dtype=source.dtype, device=device)
        T_ts_new[0:3, 0:3] = del_C.T @ C_ts
        T_ts_new[0:3, 3:] = r_st_t - del_r
        T_ts = T_ts_new

        # Check for convergence
        if torch.linalg.norm(del_T_ts.detach()) < tolerance:
            break

    #print("ICP converged in {} iterations".format(ii))

    # Update source point cloud
    ps_s = (T_ts[0:3, 0:3] @ ps_s.squeeze().T + T_ts[0:3, 3].reshape(3,1)).T.unsqueeze(2)

    return ps_s, T_ts

def pt2pl_dICP_single(source, target, T_init, max_iterations=100, tolerance=1e-8, trim_dist=5.0, huber_delta=1.0):
    """
    Point-to-plane ICP algorithm.
    :param source_tree: Source point cloud tree resolved in source frame ps_s [n x 6].
    :param target_tree: Target point cloud tree resolved in target frame pt_t [m x 6].
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Tolerance for convergence.
    :return: Transformed source point cloud and transformation from source to target T_ts.
    """

    # Confirm that source, target, and T_init types match
    assert source.dtype == target.dtype == T_init.dtype
    # Initialize transformation matrix, have same type as source
    T_ts = T_init

    # Source points are resolved in s frame, ignore normals for source
    if source.shape[1] == 6:
        ps_s = source[:, 0:3]
    else:
        ps_s = source

    # Iterate until convergence
    for ii in range(max_iterations):
        # Find nearest neighbour for each point in source, these points are in target frame
        err = torch.zeros(len(source), 1, dtype=source.dtype)
        w = torch.zeros(len(source), 1, dtype=source.dtype)
        J = torch.zeros(len(source), 6, dtype=source.dtype)
        nn_t = torch.zeros((source.shape[0], target.shape[1]), dtype=source.dtype)
        # Extract T_ts components
        C_ts = T_ts[0:3, 0:3]
        r_st_t = T_ts[0:3, 3:]
        for jj in range(len(source)):
            ps_s_jj = ps_s[jj, :].reshape(3, 1)

            # Transform point to best guess target frame
            ps_t_jj = C_ts @ ps_s_jj + r_st_t

            # Find nearest neighbour
            nn_jj = diff_nn_single(ps_t_jj.reshape(1, 3), target)
            nn_t[jj] = nn_jj
            nn_t_jj = nn_jj[:3].reshape(3, 1)
            nn_norm_jj = nn_jj[3:].reshape(3, 1)

            # Compute error
            nn_err = ps_t_jj - nn_t_jj
            err_jj = nn_err.T @ nn_norm_jj

            # Compute weight based on trim distance and huber delta

            # Produces a weight of 0 if the error is greater than the trim distance and 1 otherwise
            #w_jj = 0.5*torch.tanh(10*(trim_dist - torch.norm(nn_err)) - 2.0) + 0.5 #/ (trim_dist - torch.norm(nn_err))
            #w_trim = 0.5*torch.tanh((trim_dist - torch.norm(nn_err)) - 3.0) + 0.5
            steep_fact = 5.0
            if huber_delta not in [0.0, None]:
                #w_huber = torch.where(torch.abs(err_jj) > huber_delta, huber_delta / torch.abs(err_jj), 1.0)
                # Use pseudo huber loss
                w_huber = huber_delta**2 / (huber_delta**2 + err_jj**2)
            else:
                w_huber = torch.ones_like(err)

            if trim_dist not in [0.0, None]:
                #w_jj = torch.where(torch.norm(nn_err) > trim_dist, torch.tensor(0.0), w_huber)
                w_trim = 0.5*torch.tanh(steep_fact*(trim_dist - torch.norm(nn_err)) - 3.0) + 0.5
            else:
                w_trim = torch.ones_like(err)

            w_jj = w_huber * w_trim
            
            #w_jj = 1.0

            #w_trim = 0.5*torch.tanh((trim_dist - torch.norm(nn_err)) - 3.0) + 0.5

            # Compute Jacobian component of err_jj with respect to T_ts
            J_C = nn_norm_jj.T @ skew_operator_single(C_ts @ ps_s_jj)
            J_r = - nn_norm_jj.T

            # Insert components into matrices
            err[jj] = err_jj
            w[jj] = w_jj
            J[jj, 0:3] = J_C
            J[jj, 3:6] = J_r
        
        # Assemble weight matrix
        W = torch.diag(w.reshape(-1))

        A = J.T @ W @ J + 1e-12 * torch.eye(6, dtype=source.dtype)

        del_T_ts = - torch.linalg.inv(A) @ J.T @ W @ err

        # Update T_ts
        del_C = torch.matrix_exp(skew_operator_single(del_T_ts[0:3, 0]))
        del_r = del_T_ts[3:6, :]
        T_ts_new = torch.eye(4, dtype=source.dtype)
        T_ts_new[0:3, 0:3] = del_C.T @ C_ts
        T_ts_new[0:3, 3:] = r_st_t - del_r
        T_ts = T_ts_new

        # Check for convergence
        #print(torch.linalg.norm(del_T_ts.detach()))
        if torch.linalg.norm(del_T_ts.detach()) < tolerance:
            break

    #print("ICP converged in {} iterations".format(ii))

    # Update source point cloud
    ps_s = (T_ts[0:3, 0:3] @ ps_s.T + T_ts[0:3, 3].reshape(3,1)).T

    return ps_s, T_ts

def skew_operator(x):
    """
    Skew operator for a 3x1 vector.
    :param x: A nx3x1 tensor.
    :return: Skew operator for each 3x1 vector in the input tensor.
    """
    # Expand x to be 1x3nx1 if n=1
    if x.shape[0] == 3:
        x = x.reshape(1, 3, 1)

    # Transpose the tensor to get dimensions of 1x3xn
    x = x.transpose(1, 2)

    # Extract x, y, and z components of the tensor
    x_comp = x[:, :, 0]
    y_comp = x[:, :, 1]
    z_comp = x[:, :, 2]
    
    # Compute the skew-symmetric matrices for each vector
    skew_mat = torch.stack([torch.zeros_like(x_comp), -z_comp, y_comp, z_comp, torch.zeros_like(y_comp), -x_comp, 
                            -y_comp, x_comp, torch.zeros_like(z_comp)], dim=2).reshape(-1, 3, 3)
    
    return skew_mat

def skew_operator_single(x):
    """
    Skew operator for a 3x1 vector.
    :param x: A 3x1 vector.
    :return: Skew operator.
    """
    return torch.tensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

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

    plt.xlim(-4, 6)
    plt.ylim(-2, 10)
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

    plt.xlim(-4, 6)
    plt.ylim(-2, 10)
    plt.show()