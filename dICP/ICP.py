""" 
    Various ICP algorithms for point clouds.
"""

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from dICP.KDTree_ICP import KDTree
from dICP.nn import nn
from dICP.loss import loss
import torch.nn.functional as F

class ICP:
    def __init__(self, config_path='../config/dICP_config.yaml', icp_type='pt2pl', max_iterations=100, tolerance=1e-12, differentiable=True):
        def load_config(file_path):
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            return config

        # Load in config data from desired config path
        self.config = load_config(config_path)

        # Set up ICP parameters
        self.icp_type = icp_type
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.const_iter = self.config['dICP']['parameters']['const_iter']
        self.verbose = self.config['dICP']['logging']['verbose']
        self.target_pad_val = self.config['dICP']['parameters']['target_pad_val']
        self.diff = differentiable

        self.nn = nn(self.diff)

    def icp(self, source, target, T_init, trim_dist=None, loss_fn=None, dim=3):
        return self.dICP(source, target, T_init, trim_dist, loss_fn, dim)
        #return self.dICP_single(source, target, T_init, trim_dist, loss_fn, dim)
    
        if self.diff:
            return self.dICP(source, target, T_init, trim_dist, huber_delta, dim)
        else:
            if self.icp_type == 'pt2pt':
                return self.pt2pt_ICP(source, target, T_init)
            elif self.icp_type == 'pt2pl':
                pass
                #self.pt2pl_ICP(source, target, T_init, trim_dist, huber_delta, dim)

    def pt2pt_kdICP(self, source_tree, target_tree, T_init):
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

    def pt2pt_dICP_SVD(self, source, target, T_init, trim_dist=None, huber_delta=None, dim=3):
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
        ps_s = source[:, 0:3]
        target = target[:, 0:3]

        # Iterate until convergence
        for ii in range(self.max_iterations):
            # Find nearest neighbour for each point in source, these points are in target frame
            nn_t = torch.zeros((source.shape[0], target.shape[1]), dtype=source.dtype)
            for jj in range(len(source)):
                nn_t[jj] = self.nn.find_nn(ps_s[jj].reshape(1, 3), target)

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

        if self.verbose:
            print("ICP converged in {} iterations".format(ii))

        return ps_s, T_ts

    def dICP(self, source, target, T_init, trim_dist=None, loss_fn=None, dim=3):
        """
        Point-to-plane ICP algorithm. Note, source and target pointclouds at each
        batch may be of different sizes. This is dealth with.
        :param source: Source point cloud resolved in source frame ps_s (n, 3/6)
                        (only first 3 dimensions matter). For batch operations, 
                        this can be a tensor of shape (N, n, 6) or a list of
                        length N of tensors of shape (n_N, 3/6).
        :param target: Target point cloud resolved in target frame pt_t (m, 3/6).
                        (only first 3 dimensions matter for pt2pt). For batch operations,
                        this can be a tensor of shape (N, m, 6) or a list of
                        length N of tensors of shape (m_N, 3/6).
        :param T_init: Initial transformation from source to target T_ts (4, 4). For batch
                        operations, this can be a tensor of shape (N, 4, 4) or a list of
                        length N of tensors of shape (4, 4).
        :param trim_dist: Distance threshold for trimming outliers. If None, no trimming is done.
        :param loss_fn: Loss function to use in the form 
                        loss_fn = {"name": "[huber/cauchy]", "metric": FLOAT_NUMBER}
                        If None, no loss function used.
        :return: Transformed source point cloud and transformation from source to target T_ts.
        """
        # Handle batch sizing for various possible inputs
        # This also trims the source pointcloud to dim 3
        # source is now a tensor of shape (N, n, 3)
        # target is now a tensor of shape (N, m, 3)
        # T_init is now a tensor of shape (N, 4, 4)
        source, target, T_init, w_init = self.batch_size_handling(source, target, T_init)
        N = source.shape[0]

        # Confirm that source, target, and T_init types match
        assert source.dtype == target.dtype == T_init.dtype

        #source = source.squeeze(0)
        #target = target.squeeze(0)
        #T_init = T_init.squeeze(0)
        #w_init = w_init.squeeze(0)

        if self.icp_type == 'pt2pl':
            # Confirm that normals for target exist
            assert target.shape[2] == 6
        else:
            target = target[:, :, :3]

        # Load in param
        steep_fact = self.config['dICP']['parameters']['tanh_steepness']

        # Save device type so that everything can be on same device
        device = source.device

        # Initialize transformation matrix
        C_ts = T_init[:, 0:3, 0:3]
        r_st_t = T_init[:, 0:3, 3:]

        # Source points are resolved in s frame, ignore normals for source (if they exist)
        ps_s = source.transpose(1, 2)

        # Iterate until convergence
        for ii in range(self.max_iterations):
            # C_ts is shape (N, 3, 3)
            # r_st_t is shape (N, 3, 1)
            # ps_s is shape (N, n, 3)
            # Transform points to best guess target frame
            ps_t = C_ts @ ps_s + r_st_t

            # Find nearest neighbours, these points are in target frame
            nn = self.nn.find_nn(ps_t, target).transpose(1, 2)
            nn_t = nn[:, :3, :]

            # Compute errors
            if self.icp_type == 'pt2pl':
                nn_err = (ps_t - nn_t).transpose(1,2)
                nn_norm = nn[:, 3:].transpose(1,2)
                err = torch.sum(nn_err * nn_norm, axis=2).unsqueeze(-1)
            else:
                nn_err = (ps_t - nn_t).transpose(1,2).reshape(N, -1, 1)
                err = nn_err

            # Compute weights based on trim distance and huber delta
            # Note, weights are initialized during the batch setup process
            w = w_init
            if trim_dist is not None and trim_dist >= 0.0:
                trim = loss(name="trim", metric=trim_dist, differentiable=self.diff, tanh_steepness=steep_fact)
                w = w * trim.get_weight(nn_err)

            if loss_fn is not None:
                loss_fn_obj = loss(name=loss_fn['name'], metric=loss_fn['metric'], differentiable=self.diff, tanh_steepness=steep_fact)
                w = w * loss_fn_obj.get_weight(err).squeeze(-1)

            # Compute Jacobian components of err with respect to T_ts
            if self.icp_type == 'pt2pl':
                skew_input = (C_ts @ ps_s).transpose(1,2)
                nn_norm_prep = nn_norm.unsqueeze(-1)
                J_C = (self.__skew_operator(skew_input).transpose(2,3) @ nn_norm_prep).squeeze(-1)
                J_r = - nn_norm
            else:
                skew_input = (C_ts @ ps_s).transpose(1,2)
                J_C = self.__skew_operator(skew_input).view(N, -1, 3)
                J_r = - torch.eye(3, device=device).repeat(N, skew_input.shape[1], 1)

            # Combine Jacobian components into a single Jacobian matrix
            J = torch.cat((J_C, J_r), dim=2)

            # If only 2D, zero out all 3D components of gradient
            if dim == 2:
                D = torch.zeros((6,3), dtype=source.dtype, device=device)
                D[2,0] = D[3,1] = D[4,2] = 1.0
                J = J @ D
            
            # Compute weighted Jacobian and error, this avoids having to form a weight matrix
            # which can be memory expensive
            w_sqrt = torch.sqrt(w)
            err_w = w_sqrt.unsqueeze(-1) * err
            J_w = w_sqrt.unsqueeze(-1) * J

            # Compute update
            J_w_T = J_w.transpose(1,2)
            A = J_w_T @ J_w + 1e-12 * torch.eye(J.shape[2], dtype=source.dtype, device=device)
            del_T_ts = - torch.linalg.inv(A) @ J_w_T @ err_w

            # If only 2D, del_T_ts will only have 3 components, update them accordingly
            if dim == 2:
                temp_step = torch.zeros((N, 6, 1), dtype=source.dtype, device=device)
                temp_step[:, 2:5] = del_T_ts
                del_T_ts = temp_step

            # Isolate update rotation/translation components
            del_C = torch.matrix_exp(self.__skew_operator(del_T_ts[:,0:3].transpose(1,2)).squeeze(1))
            del_r = del_T_ts[:,3:6]

            # Update T_ts
            C_ts_new = del_C.transpose(1,2) @ C_ts
            C_ts = C_ts_new
            r_st_t_new = r_st_t - del_r
            r_st_t = r_st_t_new

            # Check for convergence if not constant iterations
            # TODO: Do better convergence check per N dimension
            if torch.linalg.norm(del_T_ts.detach()) < self.tolerance and not self.const_iter:
                break

        if self.verbose:
            print("ICP converged in {} iterations".format(ii+1))
            print("Final del_T_ts: {}".format(torch.linalg.norm(del_T_ts.detach())))

        # Update source point cloud with converged transformation
        ps_t_final = (C_ts @ ps_s + r_st_t).transpose(1,2)

        # Form final transformation
        T_ts_ones = torch.ones((N, 4), dtype=source.dtype, device=device)
        T_ts = torch.diag_embed(T_ts_ones)
        T_ts[:, 0:3, 0:3] = C_ts
        T_ts[:, 0:3, 3] = r_st_t.squeeze(-1)

        return ps_t_final, T_ts

    def batch_size_handling(self, source, target, T_init):
        """
        Properly transforms inputs to do ICP for entire batch at once. Note, source and
        target pointclouds at each batch may be different. This is dealt with.
        :param source: Source point cloud (n x 3) or (N x n x 3) or list len(N) (n_N x 3).
        :param target: Target point cloud (m x 3) or (N x m x 3) or list len(N) (m_N x 3).
        :param T_init: Initial transformation (4 x 4) or (N x 4 x 4) or list len(N) (4 x 4).
        """
        # Deal with source. If source has 6 columns (x,y,z,norm_x,norm_y,norm_z), remove normals
        # Also want to return a weight vector for each point in source, with a weight of 1 if
        # the point is not a padded point and 0 if it is a padded point
        
        pt_device = source[0].device
        pt_dtype = source[0].dtype

        # First, handle case where source is list of (n_i x 3) tensors
        if isinstance(source, list):
            source_batch = source[0][:,:3].unsqueeze(0)
            w = torch.ones((1, source_batch.shape[1]), dtype=pt_dtype, device=pt_device)
            for source_i in source[1:]:
                if len(source_i.shape) != 2 or (source_i.shape[1] != 3 and source_i.shape[1] != 6):
                    raise ValueError("source list must contain (n x 3/6) tensors")
                # If source_i has less points than max points in batch, pad with zeros and add
                if source_i.shape[0] < source_batch.shape[1]:
                    # Compute weight vector
                    w_i_ones = torch.ones(source_i.shape[0], dtype=pt_dtype, device=pt_device)
                    w_i_zeros = torch.zeros(source_batch.shape[1] - source_i.shape[0], dtype=pt_dtype, device=pt_device)
                    w_i = torch.hstack((w_i_ones, w_i_zeros))
                    # Update source_i with zeros
                    source_i = torch.vstack((source_i[:,:3], torch.zeros((source_batch.shape[1] - source_i.shape[0], 3), dtype=pt_dtype, device=pt_device)))
                    source_batch = torch.vstack((source_batch, source_i.unsqueeze(0)))
                # If source_i has more points than max points in batch, pad batch with zeros and add
                elif source_i.shape[0] > source_batch.shape[1]:
                    # Compute weight vector
                    w_i = torch.ones(source_i.shape[0], dtype=pt_dtype, device=pt_device)
                    # Update w with zeros
                    w = torch.hstack((w, torch.zeros((w.shape[0], source_i.shape[0] - source_batch.shape[1]), dtype=pt_dtype, device=pt_device) ))
                    # Update source_batch with zeros
                    source_batch_zeros = torch.zeros((source_batch.shape[0], source_i.shape[0] - source_batch.shape[1], source_batch.shape[2]), dtype=pt_dtype, device=pt_device)
                    source_batch = torch.hstack((source_batch, source_batch_zeros))
                    # Add source_i to source_batch
                    source_batch = torch.vstack((source_batch, source_i[:, :3].unsqueeze(0)))
                # If source_i has same number of points as max points in batch, just add
                else:
                    source_batch = torch.vstack((source_batch, source_i[:, :3].unsqueeze(0)))
                    w_i = torch.ones(source_i.shape[0], dtype=pt_dtype, device=pt_device)
                w = torch.cat((w, w_i.unsqueeze(0)), dim=0)        
        # Next, handle case where source is (n x 3)
        elif len(source.shape) == 2 and (source.shape[1] == 3 or source.shape[1] == 6):
            source_batch = source[:, :3].unsqueeze(0)
            w = torch.ones((1, source_batch.shape[1]), dtype=pt_dtype, device=pt_device)
        # Finally, handle case where source is (N x n x 3)
        elif len(source.shape) == 3 and (source.shape[2] == 3 or source.shape[2] == 6):
            source_batch = source[:, :, :3]
            w = torch.ones((source_batch.shape[0], source_batch.shape[1]), dtype=pt_dtype, device=pt_device)
        else:
            raise ValueError("source must be (n x 3/6) or (N x n x 3/6) or list len(N) (n_N x 3/6)")
        
        # Deal with target. For target, we want to pad with value large enough to never get
        # selected as closest point. Take the max of the source point cloud and multiply by target_pad_val
        # First, handle case where target is list of (m_i x 3/6) tensors
        if isinstance(target, list):
            target_batch = target[0].unsqueeze(0)
            target_dim = target[0].shape[1]
            target_pad = torch.max(source_batch) * self.target_pad_val
            for target_i in target[1:]:
                if len(target_i.shape) != 2 or target_i.shape[1] != target_dim:
                    raise ValueError("target list must contain (m x 3/6) tensors. All tensors must have same number of columns")
                # If target_i has less points than max points in batch, pad with zeros and add
                if target_i.shape[0] < target_batch.shape[1]:
                    pad_i = target_pad*torch.ones((target_batch.shape[1] - target_i.shape[0], 3), dtype=pt_dtype, device=pt_device)
                    target_i = torch.vstack((target_i, pad_i))
                    target_batch = torch.vstack((target_batch, target_i.unsqueeze(0)))
                # If target_i has more points than max points in batch, pad batch with zeros and add
                elif target_i.shape[0] > target_batch.shape[1]:
                    pad_i = target_pad*torch.ones((target_batch.shape[0], target_i.shape[0] - target_batch.shape[1], target_batch.shape[2]), dtype=pt_dtype, device=pt_device)
                    target_batch = torch.hstack((target_batch, pad_i))
                    target_batch = torch.vstack((target_batch, target_i.unsqueeze(0)))
                # If target_i has same number of points as max points in batch, just add
                else:
                    target_batch = torch.vstack((target_batch, target_i.unsqueeze(0)))        
        # Next, handle case where target is (m x 3/6)
        elif len(target.shape) == 2 and (target.shape[1] == 3 or target.shape[1] == 6):
            target_batch = target.unsqueeze(0)

        # Finally, handle case where target is (N x m x 3/6)
        elif len(target.shape) == 3 and (target.shape[2] == 3 or target.shape[2] == 6):
            target_batch = target
        else:
            raise ValueError("target must be (m x 3/6) or (N x m x 3/6) or list len(N) (m_N x 3/6)")

        # Deal with T_init
        if isinstance(T_init, list):
            T_init_batch = torch.stack(T_init, dim=0)
        elif T_init.shape == (4,4):
            T_init_batch = T_init.unsqueeze(0)
        elif len(T_init.shape) == 3 and T_init.shape[1:] == (4,4):
            T_init_batch = T_init
        else:
            raise ValueError("T_init must be (4 x 4) or (N x 4 x 4) or list len(N) (4 x 4)")


        # As a last step, fix the dimensions of the weight. If pt2pt, then each point will have
        # a 3x1 error. If pt2pl, then each point will have a 1x1 error.
        if self.icp_type == 'pt2pt':
            w = w.repeat_interleave(3, dim=1)

        return source_batch, target_batch, T_init_batch, w

    def __skew_operator(self, x):
        """
        Batch skew operator.
        :param x: Input tensor (N, n, 3).
        :return: Output skew matrix (N, n, 3, 3).
        """

        # Extract x, y, and z components of the tensor
        x_comp = x[:, :, 0]
        y_comp = x[:, :, 1]
        z_comp = x[:, :, 2]
        
        # Compute the skew-symmetric matrices for each vector
        skew_mat = torch.stack([torch.zeros_like(x_comp), -z_comp, y_comp,
                                z_comp, torch.zeros_like(y_comp), -x_comp, 
                                -y_comp, x_comp, torch.zeros_like(z_comp)], dim=2
                                ).view(x.shape[0], x.shape[1], 3, 3)

        return skew_mat