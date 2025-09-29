import numpy as np
import os

class Weight:
    """
    Weight manager using an empirically generated manifold reference model.
    This version uses ABSOLUTE neighbor counts for comparison.
    """

    def __init__(self, dim: int):
        """
        Initialize Weight object using a pre-computed reference model.

        Parameters:
        - dim (int): The maximum topological dimension to calculate weights for.
        """
        self.dim = dim
        self.weights = {i: 0.0 for i in range(dim + 1)}

        # --- CORRECTED: Load the empirical reference model ---
        model_path = 'manifold_reference_model.npy'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"'{model_path}' not found. Please run 'generate_reference_model.py' first.")
        
        reference_model = np.load(model_path, allow_pickle=True).item()

        # --- CORRECTED: Use the absolute counts from the model directly ---
        # No normalization is performed. This compares absolute measured counts
        # to absolute reference counts from ideal manifolds.
        bounds = [reference_model.get(m, 0) for m in range(dim + 1)]
        
        self.sets = np.full((len(bounds), 3), -1.0)
        for i in range(dim + 1):
            self.sets[i, 1] = float(bounds[i])
            if i > 0:
                self.sets[i, 0] = float(bounds[i - 1])
            if i < dim:
                self.sets[i, 2] = float(bounds[i + 1])

    @staticmethod
    def linear_gen(x: float, center: float, left: float, right: float):
        """
        Triangular (hat) basis function.
        """
        influence: float = 0.0
        if left != -1 and right != -1 and not (left <= center <= right):
            return 0.0
        # Avoid division by zero
        if left != -1 and abs(center - left) < 1e-9:
            return 1.0 if abs(x - center) < 1e-9 else 0.0
        if right != -1 and abs(right - center) < 1e-9:
            return 1.0 if abs(x - center) < 1e-9 else 0.0

        if left == -1 and right != -1:
            if center <= x <= right: influence = (right - x) / (right - center)
        elif right == -1 and left != -1:
            if left <= x <= center: influence = (x - left) / (center - left)
        elif left != -1 and right != -1:
            if left <= x <= center: influence = (x - left) / (center - left)
            elif center < x <= right: influence = (right - x) / (right - center)
        return max(0.0, min(1.0, influence))

    def add_point_info(self, x: float | int, point_weight: float = 1.0):
        for i in range(self.dim + 1):
            influence = Weight.linear_gen(x, self.sets[i, 1], self.sets[i, 0], self.sets[i, 2])
            self.weights[i] += influence * point_weight  # scale by weight
    
    def __add__(self, other):
        """
        In-place addition of another Weight's weights.
        """
        for i in range(self.dim + 1):
            self.weights[i] += other.weights[i]