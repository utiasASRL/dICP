import torch
import torch.nn.functional as F

def diff_nn(x, y):
    """
    Computes the differentiable nearest neighbor of x from the matrix y using softmax.
    """
    
    # Expand x and y to have an additional dimension for broadcasting
    x_use = x.unsqueeze(1)  # shape: (m, 1, 3)
    y_use = y.unsqueeze(0)  # shape: (1, n, 3/6)

    # If y has 6 elements, then normals are included, in this case extract first 3 for operations
    # Compute the squared Euclidean distances between x and each point in y
    distances = torch.sum((x_use - y_use[:,:,:3])**2, dim=2)     # shape: (m, n)

    # Apply the softmax function to the negative distances to obtain a probability distribution
    probs = F.softmax(-distances, dim=1)    # shape: (m, n)

    # Compute the argmax of the probability distribution to obtain the index of the closest point
    index = torch.argmax(probs, dim=1)  # shape: (m,)

    # Select the closest point from y using the index
    neighbors = torch.gather(y_use, 1, index.unsqueeze(0).unsqueeze(2).repeat(1, 1, 6)).squeeze(1)

    return neighbors

def diff_nn_single(x, y):
    """
    Computes the differentiable nearest neighbor of x from the matrix y using softmax.
    """
    
    # If y has 6 elements, then normals are included, in this case extract first 3 for operations
    # Compute the squared Euclidean distances between x and each point in y
    distances = torch.sum((x - y[:, :3])**2, dim=1)

    # Apply the softmax function to the negative distances to obtain a probability distribution
    probs = F.softmax(-distances, dim=0)

    # Compute the argmax of the probability distribution to obtain the index of the closest point
    index = torch.argmax(probs)

    # Select the closest point from y using the index
    neighbor = y[index]

    return neighbor


def diff_nn2(x, y):
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