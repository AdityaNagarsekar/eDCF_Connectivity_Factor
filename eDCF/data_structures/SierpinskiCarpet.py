# Import statements
from typing import Dict, Tuple, Any
import numpy as np
from scipy.spatial import cKDTree


class SierpinskiCarpet:
    """
    Generates Sierpinski Carpet points and stores non-carpet points in Mask.npy.
    """

    def __init__(
        self,
        identity: int,
        iterations: int = 1000000,
        mask_points: int = 1000000,
        min_distance: float = 0.01,
        parameters: Dict[str, Any] = None,
    ):
        """
        Constructor for SierpinskiCarpet

        :param identity: Unique identifier for the Sierpinski Carpet instance (identity > 0)
        :param iterations: Number of points to generate for the Sierpinski Carpet
        :param mask_points: Number of non-carpet points to generate
        :param min_distance: Minimum distance from any carpet point to consider a non-carpet point valid
        :param parameters: Dictionary of parameters for Sierpinski Carpet generation
        """
        self.__identity = identity  # Identity assignment
        self.__iterations = iterations  # Number of carpet iterations
        self.__non_carpet_points = mask_points  # Number of non-carpet points
        self.__min_distance = min_distance  # Minimum distance to avoid overlap
        self.__parameters = parameters if parameters else self.__default_parameters()
        self.__datapoints: np.ndarray = np.empty((self.__iterations, 2), float)  # Carpet datapoints
        self.__non_carpet_datapoints: np.ndarray = np.empty((0, 2), float)  # Non-carpet datapoints

    @staticmethod
    def __default_parameters() -> Dict[str, Any]:
        """
        Defines the default parameters for the Sierpinski Carpet.

        :return: Dictionary of parameters
        """
        # Define the transformations for the chaos game method
        # There are 8 transformations corresponding to the 8 outer squares
        scale = 1 / 3
        return {
            "transformations": [
                {"a": scale, "b": 0.0,   "c": 0.0,   "d": scale, "e": 0.0,       "f": 0.0},
                {"a": scale, "b": 0.0,   "c": 0.0,   "d": scale, "e": scale,     "f": 0.0},
                {"a": scale, "b": 0.0,   "c": 0.0,   "d": scale, "e": 2*scale,   "f": 0.0},
                {"a": scale, "b": 0.0,   "c": 0.0,   "d": scale, "e": 0.0,       "f": scale},
                # Skip the center square
                {"a": scale, "b": 0.0,   "c": 0.0,   "d": scale, "e": 2*scale,   "f": scale},
                {"a": scale, "b": 0.0,   "c": 0.0,   "d": scale, "e": 0.0,       "f": 2*scale},
                {"a": scale, "b": 0.0,   "c": 0.0,   "d": scale, "e": scale,     "f": 2*scale},
                {"a": scale, "b": 0.0,   "c": 0.0,   "d": scale, "e": 2*scale,   "f": 2*scale},
            ],
            "probabilities": [1/8] * 8,  # Equal probability for each transformation
        }

    # Getter methods

    def get_identity(self) -> int:
        """
        :return: Identity of the Sierpinski Carpet
        """
        return self.__identity

    def get_datapoints(self) -> np.ndarray:
        """
        :return: Generated Sierpinski Carpet datapoints as a 2D NumPy array
        """
        return self.__datapoints

    def get_non_carpet_datapoints(self) -> np.ndarray:
        """
        :return: Generated non-carpet datapoints as a 2D NumPy array
        """
        return self.__non_carpet_datapoints

    # Getter methods end

    def generate(self, n: int) -> np.ndarray:
        """
        Generates the Sierpinski Carpet and random non-carpet points within the set's bounding box,
        ensuring they do not overlap with any carpet datapoints, and stores them in Mask.npy.

        :return: 2D NumPy array of generated Sierpinski Carpet points (x, y)
        """
        params = self.__parameters
        transformations = params["transformations"]
        probabilities = params["probabilities"]
        cumulative_probs = np.cumsum(probabilities)
        cumulative_probs[-1] = 1.0  # Ensure the last probability sums to 1

        x, y = 0.5, 0.5  # Starting point (center of the initial square)

        # Generate Sierpinski Carpet datapoints using the Chaos Game method
        for i in range(self.__iterations):
            r = np.random.random()
            for j, trans in enumerate(transformations):
                if r <= cumulative_probs[j]:
                    a, b, c, d, e, f = (
                        trans["a"],
                        trans["b"],
                        trans["c"],
                        trans["d"],
                        trans["e"],
                        trans["f"],
                    )
                    x_new = a * x + b * y + e
                    y_new = c * x + d * y + f
                    x, y = x_new, y_new
                    break
            self.__datapoints[i, 0] = x
            self.__datapoints[i, 1] = y

        # Generate non-carpet data
        carpet_points_data = self.get_datapoints()
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0

        # Build a KD-Tree for efficient nearest neighbor search
        tree = cKDTree(carpet_points_data)

        # Initialize list to hold valid non-carpet points
        valid_non_carpet = []

        # Define batch size for generating points
        batch_size = 100000  # Adjust based on memory and performance considerations

        while len(valid_non_carpet) < self.__non_carpet_points:
            remaining = self.__non_carpet_points - len(valid_non_carpet)
            current_batch_size = min(batch_size, remaining * 2)  # Generate extra to account for rejections

            # Generate random points
            rand_x = np.random.uniform(x_min, x_max, current_batch_size)
            rand_y = np.random.uniform(y_min, y_max, current_batch_size)
            candidates = np.vstack((rand_x, rand_y)).T

            # Query the KD-Tree for points within min_distance
            distances, _ = tree.query(candidates, k=1)
            mask = distances >= self.__min_distance

            valid_candidates = candidates[mask]

            # Add valid candidates to the list
            for point in valid_candidates:
                if len(valid_non_carpet) < self.__non_carpet_points:
                    valid_non_carpet.append(point)
                else:
                    break

        self.__non_carpet_datapoints = np.array(valid_non_carpet)
        np.save("data_structures/Mask.npy", self.__non_carpet_datapoints)

        labels = np.full((len(self.__datapoints), 1), self.__identity)
        self.__datapoints = np.concatenate((self.__datapoints, labels), axis=1)

        return self.__datapoints

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Computes the bounding box of the Sierpinski Carpet datapoints.

        :return: Tuple containing (x_min, x_max, y_min, y_max)
        """
        if self.__datapoints.size == 0:
            raise ValueError("Datapoints not generated. Call generate() first.")
        x_min, x_max = self.__datapoints[:, 0].min(), self.__datapoints[:, 0].max()
        y_min, y_max = self.__datapoints[:, 1].min(), self.__datapoints[:, 1].max()
        return x_min, x_max, y_min, y_max
