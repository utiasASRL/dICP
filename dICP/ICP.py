""" 
    Various ICP algorithms for point clouds.
"""

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from dICP.kd_nn import KDTree
from dICP.diff_nn import diff_nn
import torch.nn.functional as F

class ICP:
    def __init__(self, icp_type='pt2pl', max_iterations=100, tolerance=1e-12, differentiable=None):
        def load_config(file_path):
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            return config

        # Load in config data from desired config path
        config = load_config('../config/dICP_config.yaml')
        config_path = config['setup']['config_path']
        self.config = load_config(config_path)

        # Set up ICP parameters
        self.icp_type = icp_type
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.verbose = self.config['dICP']['logging']['verbose']
        self.sparse = self.config['dICP']['functionality']['sparse']
        if differentiable is not None:
            self.diff = differentiable
        else:
            self.diff = self.config['dICP']['functionality']['differentiable'] 

    def icp(self, source, target, T_init, trim_dist=5.0, huber_delta=1.0, dim=3):
        if self.diff:
            if self.icp_type == 'pt2pt':
                return self.pt2pt_dICP(source, target, T_init)
            elif self.icp_type == 'pt2pl':
                return self.pt2pl_dICP(source, target, T_init, trim_dist, huber_delta, dim)
        
        else:
            if self.icp_type == 'pt2pt':
                return self.pt2pt_ICP(source, target, T_init)
            elif self.icp_type == 'pt2pl':
                pass
                #self.pt2pl_ICP(source, target, T_init, trim_dist, huber_delta, dim)


    def pt2pt_ICP(self, source_tree, target_tree, T_init):
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
        for ii in range(self.max_iterations):
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
            if np.sum((ps_s - nn_t) ** 2) < self.tolerance:
                break

        print("ii: ", ii)

        return KDTree(ps_s.tolist()), T_ts

    def pt2pt_dICP(self, source, target, T_init):
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
        for ii in range(self.max_iterations):
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
            if torch.sum((ps_s - nn_t) ** 2) < self.tolerance:
                break

        return ps_s, T_ts


    def pt2pl_dICP(self, source, target, T_init, trim_dist=5.0, huber_delta=1.0, dim=3):
        """
        Point-to-plane ICP algorithm.
        :param source_tree: Source point cloud tree resolved in source frame ps_s [n x 6].
        :param target_tree: Target point cloud tree resolved in target frame pt_t [m x 6].
        :param max_iterations: Maximum number of iterations.
        :param tolerance: Tolerance for convergence.
        :return: Transformed source point cloud and transformation from source to target T_ts.
        """
        steep_fact = self.config['dICP']['parameters']['tanh_steepness']
        # Confirm that source, target, and T_init types match
        assert source.dtype == target.dtype == T_init.dtype

        # Save device type so that everything can be on same device
        device = source.device

        # Initialize transformation matrix
        T_ts = T_init

        # Source points are resolved in s frame, ignore normals for source (if they exist)
        ps_s = source[:, 0:3]

        # Iterate until convergence
        for ii in range(self.max_iterations):
            # Extract T_ts components
            C_ts = T_ts[0:3, 0:3]
            r_st_t = T_ts[0:3, 3:]
            ps_s = ps_s.reshape(-1, 3, 1)

            # Transform points to best guess target frame
            ps_t = C_ts @ ps_s + r_st_t.reshape(1, 3, 1)

            # Find nearest neighbours, these points are in target frame
            nn = diff_nn(ps_t.reshape(-1, 3), target).squeeze()
            nn_t = nn[:, :3].reshape(-1, 3, 1)
            nn_norm = nn[:, 3:].reshape(-1, 3, 1)

            # Compute errors
            nn_err = ps_t - nn_t
            err = torch.sum(nn_err * nn_norm, axis=(1, 2))

            # Compute weights based on trim distance and huber delta
            if huber_delta is not None and huber_delta > 0.0:
                #w_huber = torch.where(torch.abs(err) > huber_delta, huber_delta / torch.abs(err), torch.ones_like(err))
                # Use pseudo huber loss
                w_huber = huber_delta**2 / (huber_delta**2 + err**2)
            else:
                w_huber = torch.ones_like(err)

            if trim_dist is not None and trim_dist > 0.0:
                # Use a soft trim distance check, gradient of tanh controlled by steep_fact
                w_trim = 0.5 * torch.tanh(steep_fact * (trim_dist - torch.linalg.norm(nn_err, axis=1).squeeze()) - 3.0) + 0.5
            else:
                w_trim = torch.ones_like(err)
            
            # Compute resulting weight
            w = w_huber * w_trim

            # Compute Jacobian components of err with respect to T_ts
            J_C = torch.bmm(nn_norm.transpose(1,2), self.__skew_operator(C_ts @ ps_s)).transpose(1,2)
            J_r = - nn_norm

            # Reshape err, w, J_C, and J_r to be N-dimensional arrays
            err = err.reshape(-1)
            J_C = J_C.reshape(-1, 3)
            J_r = J_r.reshape(-1, 3)

            # Combine Jacobian components into a single Jacobian matrix
            J = torch.hstack((J_C, J_r))

            # Assemble weight matrix
            if self.sparse:
                W_idx_line = torch.arange(J.shape[0], dtype=torch.long, device=device)
                W_idx = torch.vstack((W_idx_line, W_idx_line))
                W = torch.sparse_coo_tensor(indices = W_idx, values = w, size=(J.shape[0], J.shape[0]))
            else:
                W = torch.diag(w)

            # If only 2D, zero out all 3D components of gradient
            if dim == 2:
                D = torch.zeros((6,3), dtype=source.dtype, device=device)
                D[2,0] = D[3,1] = D[4,2] = 1.0
                J = J @ D
            
            # Compute update
            A = J.T @ W @ J + 1e-12 * torch.eye(J.shape[1], dtype=source.dtype, device=device)
            del_T_ts = - torch.linalg.inv(A) @ J.T @ W @ err

            # If only 2D, del_T_ts will only have 3 components, update them accordingly
            if dim == 2:
                temp_step = torch.zeros((6), dtype=source.dtype, device=device)
                temp_step[2:5] = del_T_ts
                del_T_ts = temp_step

            # Isolate update rotation/translation components
            del_C = torch.matrix_exp(self.__skew_operator(del_T_ts[0:3]).squeeze())
            del_r = del_T_ts[3:6].reshape(3, 1)

            # Update T_ts
            T_ts_new = torch.eye(4, dtype=source.dtype, device=device)
            T_ts_new[0:3, 0:3] = del_C.T @ C_ts
            T_ts_new[0:3, 3:] = r_st_t - del_r
            T_ts = T_ts_new

            # Check for convergence
            if torch.linalg.norm(del_T_ts.detach()) < self.tolerance:
                break

        if self.verbose:
            print("ICP converged in {} iterations".format(ii))

        # Update source point cloud with converged transformation
        ps_t_final = (T_ts[0:3, 0:3] @ ps_s.squeeze().T + T_ts[0:3, 3].reshape(3,1)).T.unsqueeze(2)

        return ps_t_final, T_ts

    def __skew_operator(self, x):
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