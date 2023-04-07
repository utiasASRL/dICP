import torch
import torch.nn.functional as F

def diff_nn(x, y):
    """
    Computes the differentiable nearest neighbor of x from the matrix y using softmax.
    """
    # Compute the squared Euclidean distances between x and each point in y
    distances = torch.sum((x - y)**2, dim=1)

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
    # Compute the squared Euclidean distances between x and each point in y
    distances = torch.sum((x - y)**2, dim=1)

    # Apply the Gumbel-Softmax trick to obtain a differentiable approximation of the argmax operation
    logits = -distances
    noise = torch.empty_like(logits).exponential_().log()  # sample from Gumbel distribution
    tau = 0.01  # temperature
    noisy_logits = (logits + noise) / tau  # divide by temperature
    probs = torch.softmax(noisy_logits, dim=0)

    # Compute the weighted average of the points in y using the probabilities
    neighbor = torch.sum(probs.view(-1, 1) * y, dim=0)

    return neighbor