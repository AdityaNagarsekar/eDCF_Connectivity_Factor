# Import statements
from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree


class JuliaSet:
    """
    Generates Julia Set points and stores non-Julia points in Mask.npy.
    """

    def __init__(
        self,
        identity: int,
        iterations: int = 1000000,
        mask_points: int = 1000000,
        min_distance: float = 0.01,
        c: complex = -0.8 + 0.156j,
    ):
        """
        Constructor for JuliaSet

        :param identity: Unique identifier for the Julia Set instance (identity > 0)
        :param iterations: Number of points to generate for the Julia Set
        :param mask_points: Number of non-Julia points to generate
        :param min_distance: Minimum distance from any Julia set point to consider a non-Julia point valid
        :param c: Complex parameter for the Julia Set formula Z = Z^2 + c
        """
        self.__datapoints = ...
        self.__identity = identity  # Identity assignment
        self.__iterations = iterations  # Number of iterations
        self.__non_julia_points = mask_points  # Number of non-Julia points
        self.__min_distance = min_distance  # Minimum distance to avoid overlap
        self.__c = c  # Complex parameter
        self._datapoints: np.ndarray = np.empty((self.__iterations, 2), float)  # Julia set datapoints
        self.__non_julia_datapoints: np.ndarray = np.empty((0, 2), float)  # Non-Julia datapoints

    # Getter methods

    def get_identity(self) -> int:
        """
        :return: Identity of the Julia Set
        """
        return self.__identity

    def get_datapoints(self) -> np.ndarray:
        """
        :return: Generated Julia Set datapoints as a 2D NumPy array
        """
        return self.__datapoints

    def get_non_julia_datapoints(self) -> np.ndarray:
        """
        :return: Generated non-Julia datapoints as a 2D NumPy array
        """
        return self.__non_julia_datapoints

    # Getter methods end

    def generate(self, n: int) -> np.ndarray:
        """
        Generates the Julia Set and non-Julia data, storing non-Julia data in Mask.npy.

        :return: 2D NumPy array of generated Julia Set points (x, y) with labels
        """
        # Generate Julia set datapoints
        x_min, x_max, y_min, y_max = -1.5, 1.5, -1.5, 1.5

        # Initialize list to store points that are in the Julia Set
        julia_points = []

        # Generate random points in the region
        num_points_generated = 0
        batch_size = 10000  # Generate points in batches to improve performance

        while num_points_generated < self.__iterations:
            remaining = self.__iterations - num_points_generated
            current_batch_size = min(batch_size, remaining)

            # Randomly sample points in the complex plane
            x = np.random.uniform(x_min, x_max, current_batch_size)
            y = np.random.uniform(y_min, y_max, current_batch_size)
            Z = x + 1j * y

            # Iterate Z = Z^2 + c and check if the point remains bounded
            mask = np.ones(current_batch_size, dtype=bool)
            Z0 = Z.copy()
            for _ in range(100):  # Number of iterations to determine boundedness
                Z[mask] = Z[mask] ** 2 + self.__c
                mask[mask] = np.abs(Z[mask]) < 2

            # Points that remained bounded are considered part of the Julia Set
            bounded_points = Z0[mask]
            julia_points.extend(bounded_points)
            num_points_generated += len(bounded_points)

        # Convert list of complex numbers to NumPy array of (x, y) coordinates
        julia_points = np.array(julia_points[:self.__iterations])
        self.__datapoints = np.column_stack((julia_points.real, julia_points.imag))

        # Generate non-Julia data
        julia_set_points = self.get_datapoints()

        # Build a KD-Tree for efficient nearest neighbor search
        tree = cKDTree(julia_set_points)

        # Initialize list to hold valid non-Julia points
        valid_non_julia = []

        # Define batch size for generating points
        batch_size = 100000  # Adjust based on memory and performance considerations

        while len(valid_non_julia) < self.__non_julia_points:
            remaining = self.__non_julia_points - len(valid_non_julia)
            current_batch_size = min(batch_size, remaining * 2)  # Generate extra to account for rejections

            # Generate random points
            rand_x = np.random.uniform(x_min, x_max, current_batch_size)
            rand_y = np.random.uniform(y_min, y_max, current_batch_size)
            candidates = np.vstack((rand_x, rand_y)).T

            # Query the KD-Tree for nearest distances
            distances, _ = tree.query(candidates, k=1)
            mask = distances >= self.__min_distance

            valid_candidates = candidates[mask]

            # Add valid candidates to the list
            for point in valid_candidates:
                if len(valid_non_julia) < self.__non_julia_points:
                    valid_non_julia.append(point)
                else:
                    break

            # Continue until required number of non-Julia points are generated

        self.__non_julia_datapoints = np.array(valid_non_julia)
        np.save("data_structures/Mask.npy", self.__non_julia_datapoints)

        # Assign labels to the Julia Set points
        labels = np.full((self.__datapoints.shape[0], 1), self.__identity)
        self.__datapoints = np.concatenate((self.__datapoints, labels), axis=1)

        return self.__datapoints

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Computes the bounding box of the Julia Set datapoints.

        :return: Tuple containing (x_min, x_max, y_min, y_max)
        """
        if self.__datapoints.size == 0:
            raise ValueError("Datapoints not generated. Call generate() first.")
        x_min, x_max = self.__datapoints[:, 0].min(), self.__datapoints[:, 0].max()
        y_min, y_max = self.__datapoints[:, 1].min(), self.__datapoints[:, 1].max()
        return x_min, x_max, y_min, y_max
