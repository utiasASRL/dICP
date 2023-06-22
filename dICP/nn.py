import torch
import torch.nn.functional as F

class nn:
    def __init__(self, differentiable=True):
        self.differentiable = differentiable

    def find_nn(self, x, y):
        if self.differentiable:
            return self.__diff_nn(x, y)
        else:
            return self.__non_diff_nn(x, y)

    def __diff_nn(self, x, y):
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

    def __non_diff_nn(self, x, y):
        # I don't think there is any difference in diff and non diff nearest neighbour
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

        # Find the index of the closest point
        index = torch.argmin(distances, dim=1)  # shape: (m,)

        # Select the closest point from y using the index
        neighbors = torch.gather(y_use, 1, index.unsqueeze(0).unsqueeze(2).repeat(1, 1, y_use.shape[2])).squeeze(1)

        return neighbors

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