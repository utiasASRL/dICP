import torch
import torch.nn.functional as F

class nn:
    def __init__(self, differentiable=True):
        self.differentiable = differentiable

    def find_nn(self, x, y):
        x_use, y_use = self.__handle_dimensions(x, y)
        #x_use, y_use = x, y
        if self.differentiable:
            return self.__diff_nn(x_use, y_use)
        else:
            return self.__non_diff_nn(x_use, y_use)

    def __diff_nn_old(self, x, y):
        """
        Computes the differentiable nearest neighbor of all entries in source x 
        to the target point cloud y using softmax.
        :param x: Source points [m x 3].
        :param y: Target points [n x 3/6].
        """
        
        # Expand x and y to have an additional dimension for broadcasting
        x_use = x.unsqueeze(1)  # shape: (m, 1, 3)
        y_use = y.unsqueeze(0)  # shape: (1, n, 3/6)

        # If y has 6 elements, then normals are included, in this case extract first 3 for operations
        # Compute the squared Euclidean distances between x and each point in y
        distances = torch.sum((x_use - y_use[:,:,:3])**2, dim=2)     # shape: (m, n)

        # Apply the softmax function to the negative distances to obtain a probability distribution
        #probs = F.softmax(-distances, dim=1)    # shape: (m, n)
        # I don't think this is needed?? Just use argmin of distance

        # Compute the argmax of the probability distribution to obtain the index of the closest point
        index = torch.argmin(distances, dim=1)  # shape: (m,)
        
        # Select the closest point from y using the index
        neighbors = torch.gather(y_use, 1, index.unsqueeze(0).unsqueeze(2).repeat(1, 1, y_use.shape[2])).squeeze(1)

        return neighbors

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
        # Apply the softmax function to the negative distances to obtain a probability distribution
        #probs = F.softmax(-distances, dim=1)    # shape: (m, n)
        # I don't think this is needed?? Just use argmin of distance

        # Compute the argmax of the probability distribution to obtain the index of the closest point
        index = torch.argmin(distances, dim=2)  # shape: (N, n)
        # Select the closest point from y using the index
        n_idx = index.unsqueeze(2).repeat(1, 1, y.shape[-1])
        neighbors = torch.gather(input=y, dim=1, index=n_idx)
        return neighbors

    def __non_diff_nn(self, x, y):
        # I dont think theres any difference between diff and non-diff...
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

        # Apply the softmax function to the negative distances to obtain a probability distribution
        #probs = F.softmax(-distances, dim=1)    # shape: (m, n)
        # I don't think this is needed?? Just use argmin of distance

        # Compute the argmax of the probability distribution to obtain the index of the closest point
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

    def __diff_nn2(self, x, y):
        """
        Computes the differentiable nearest neighbor of x from the matrix y using the Gumbel-Softmax trick.
        """
        # If y has 6 elements, then normals are included, in this case extract first 3 for operations
        # Compute the squared Euclidean distances between x and each point in y
        distances = torch.sum((x - y[:, :3])**2, dim=1)

        # Apply the Gumbel-Softmax trick to obtain a differentiable approximation of the argmax operation
        logits = -distances
        noise = torch.empty_like(logits).exponential_().log()  # sample from Gumbel distribution
        tau = 0.01  # temperature
        noisy_logits = (logits + noise) / tau  # divide by temperature
        probs = torch.softmax(noisy_logits, dim=0)

        # Compute the weighted average of the points in y using the probabilities
        neighbor = torch.sum(probs.view(-1, 1) * y, dim=0)

        return neighbor