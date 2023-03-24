"""
    KD Tree Nearest Neighbor Search
"""

import math
import matplotlib.pyplot as plt

class Node:
    def __init__(self, point, axis, left, right):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

class KDTree:
    def __init__(self, points):
        def build_tree(points, depth):
            if not points:
                # Reached leaf node
                return None
            # Define axis to split children nodes on
            axis = depth % len(points[0])
            points.sort(key=lambda point: point[axis])
            # Find median from sorted points
            median = len(points) // 2
            # All points with index < median will be in left subtree
            # All points with index > median will be in right subtree
            return Node(
                points[median],
                axis,
                build_tree(points[:median], depth + 1),
                build_tree(points[median + 1:], depth + 1)
            )
        self.root = build_tree(points, 0)

    def add_point(self, point):
        node = self.root
        while True:
            if point[node.axis] < node.point[node.axis]:
                if node.left is None:
                    node.left = Node(point, (node.axis + 1) % len(point), None, None)
                    break
                else:
                    node = node.left
            else:
                if node.right is None:
                    node.right = Node(point, (node.axis + 1) % len(point), None, None)
                    break
                else:
                    node = node.right

    def nearest_neighbour(self, point):
        self.nearest_point = None
        self.nearest_distance = None
        self._nearest_neighbour(self.root, point)
        return self.nearest_point

    def _nearest_neighbour(self, node, point):
        if node is None:
            return

        # Update nearest_point and nearest_distance if current node is closer
        distance = math.dist(node.point, point)
        if self.nearest_distance is None or distance < self.nearest_distance:
            self.nearest_point = node.point
            self.nearest_distance = distance

        # Check if we need to search left or right subtree
        if point[node.axis] < node.point[node.axis]:
            self._nearest_neighbour(node.left, point)
            if (node.point[node.axis] - point[node.axis]) ** 2 < self.nearest_distance:
                self._nearest_neighbour(node.right, point)
        else:
            self._nearest_neighbour(node.right, point)
            if (point[node.axis] - node.point[node.axis]) ** 2 < self.nearest_distance:
                self._nearest_neighbour(node.left, point)

# Define main function to test KDTree development
def main():
    points = [(5,4), (2,6), (13,3)]
    points=  [(5,4), (2,6), (13,3), (8,7), (3,1)]
    tree = KDTree(points)

    # Test nearest neighbour search for points
    query_point = (9, 4)
    expected_nearest_point = (8, 7)
    nearest_point = tree.nearest_neighbour(query_point)
    print(nearest_point)

    tree.add_point((10, 2))
    nearest_point = tree.nearest_neighbour(query_point)
    print(nearest_point)

if __name__ == '__main__':
    main()
