import torch
import torch.nn.functional as F

class nn:
    def __init__(self, differentiable=True, use_gumbel=True):
        self.differentiable = differentiable
        self.use_gumbel = use_gumbel

    def find_nn(self, x, y):
        x_use, y_use = self.__handle_dimensions(x, y)
        #x_use, y_use = x, y
        if self.differentiable:
            if self.use_gumbel:
                return self.__diff_nn_gumbel(x_use, y_use)
            else:
                return self.__diff_nn(x_use, y_use)
            
        else:
            return self.__non_diff_nn(x_use, y_use)

    def __diff_nn(self, x, y):
        """
        Computes the differentiable nearest neighbor of all entries in source x 
        to the target point cloud y using softmax.
        :param x: Source points (N, n, 3).
        :param y: Target points (N, m, 3/6).
        """
        # If y has 6 elements, then normals are included, in this case extract first 3 for operations
        # Compute the squared Euclidean distances between x and each point in y
        distances = torch.cdist(x, y[:, :, :3], p=2)

        # Compute the argmin of the distances to obtain the index of the closest point
        index = torch.argmin(distances, dim=2)  # shape: (N, n)
        # Select the closest point from y using the index
        n_idx = index.unsqueeze(2).repeat(1, 1, y.shape[-1])
        neighbors = torch.gather(input=y, dim=1, index=n_idx)

        return neighbors

    def __non_diff_nn(self, x, y):
        """
        Computes the differentiable nearest neighbor of all entries in source x 
        to the target point cloud y using softmax.
        :param x: Source points (N, n, 3).
        :param y: Target points (N, m, 3/6).
        """
        # Expand x and y to have an additional dimension for broadcasting

        # If y has 6 elements, then normals are included, in this case extract first 3 for operations
        # Compute the squared Euclidean distances between x and each point in y
        distances = torch.cdist(x, y[:, :, :3], p=2) # shape: (N, n, m)

        # Compute the argmin of the distances to obtain the index of the closest point
        index = torch.argmin(distances, dim=2)  # shape: (N, n)
        
        # Select the closest point from y using the index
        n_idx = index.unsqueeze(2).repeat(1, 1, y.shape[-1])
        neighbors = torch.gather(input=y, dim=1, index=n_idx)

        return neighbors

    def __handle_dimensions(self, x, y):
        """
        Handles the dimensions of the input point clouds.
        :param x: Source points (n, 3/6) or (3/6, n) or (N, n, 3/6) or (N, 3/6, n).
        :param y: Target points (m, 3/6) or (3/6, m) or (N, m, 3/6) or (N, 3/6, m).
        :return: x (N, n, 3) and y (N, m, 3/6) in the correct format.
        """
        # First, handle x
        if len(x.shape) == 2:
            x_use = x.unsqueeze(0)
        else:
            x_use = x
        # Weird edge case if input is (3, 3) or (6, 3)... no great way to handle this but assuming
        # its extremely rare
        if x_use.shape[-2] == 3 or (x_use.shape[-2] == 6 and (x_use.shape[-2] < x_use.shape[-1])):
            x_use = x[:,:3,:].transpose(1, 2)  # shape: (N, n, 3)

        assert x_use.shape[2] == 3, "x must have 3 elements in the second dimension."
        
        # Second, handle y
        if len(y.shape) == 2:
            y_use = y.unsqueeze(0)
        else:
            y_use = y
        # Weird edge case if input is (3, 3) or (6, 3)... no great way to handle this but assuming
        # its extremely rare
        if y_use.shape[-2] == 3 or (y_use.shape[-2] == 6 and (y_use.shape[-2] < y_use.shape[-1])):
            y_use = y_use.transpose(1, 2)

        assert y_use.shape[2] == 3 or y_use.shape[2] == 6, "y must have 3 or 6 elements in the second dimension."
        
        return x_use, y_use

    # Nearest neighbour using Gumbel-Softmax trick, not yet integrated
    def __diff_nn_gumbel(self, x, y):
        """
        Computes the differentiable nearest neighbor of all entries in source x 
        to the target point cloud y using softmax.
        :param x: Source points (N, n, 3).
        :param y: Target points (N, m, 3/6).
        """
        # Expand x and y to have an additional dimension for broadcasting
        x_use = x.unsqueeze(2)  # shape: (N, n, 1, 3)
        y_use = y.unsqueeze(1)  # shape: (N, 1, m, 3/6)

        # If y has 6 elements, then normals are included, in this case extract first 3 for operations
        # Compute the squared Euclidean distances between x and each point in y
        distances = torch.sum((x_use - y_use[:,:,:,:3])**2, dim=3)     # shape: (N, n, m)

        # Apply the Gumbel-Softmax trick to obtain a differentiable approximation of the argmax operation
        logits = -distances
        U = torch.rand(logits.shape, device=logits.device)  # sample from uniform distribution
        eps = 1e-20
        noise = -torch.log(-torch.log(U + eps) + eps)  # sample from Gumbel distribution
        tau = 0.1  # temperature
        noisy_logits = (logits + noise) / tau  # divide by temperature, shape: (N, n, m)
        probs = torch.softmax(noisy_logits, dim=2)  # shape: (N, n, m)
        
        # Compute the weighted average of the points in y using the probabilities
        neighbor = probs @ y

        return neighbor