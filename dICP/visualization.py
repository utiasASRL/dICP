import matplotlib.pyplot as plt
import torch


def plot_overlay(pc1, pc2, c1='b', c2='r', file_name="overlay.png"):
    # Check if points are torch tensors, if so then convert to numpy
    if isinstance(pc1, torch.Tensor):
        pc1 = pc1.detach().numpy().clone()
    if isinstance(pc2, torch.Tensor):
        pc2 = pc2.detach().numpy().clone()

    plt.figure()
    plt.scatter(pc1[:, 0], pc1[:, 1], s=0.5, c=c1)
    plt.scatter(pc2[:, 0], pc2[:, 1], s=0.5, c=c2)
    plt.savefig(file_name)

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