# Import statements
from typing import List, Dict, Tuple
import numpy as np
import random
from scipy.spatial import cKDTree


class BarnsleyFern:
    """
    Generates Barnsley Fern points and stores non-fern points in Mask.npy.
    """

    def __init__(
        self,
        identity: int,
        iterations: int = 1000000,
        mask_points: int = 1000000,
        min_distance: float = 0.01,
        transformations: List[Dict[str, float]] = None,
    ):
        """
        Constructor for BarnsleyFern

        :param identity: Unique identifier for the Barnsley Fern instance (identity > 0)
        :param iterations: Number of points to generate for the Fern
        :param mask_points: Number of non-fern points to generate
        :param min_distance: Minimum distance from any fern point to consider a non-fern point valid
        :param transformations: List of transformation dictionaries with keys 'a', 'b', 'c', 'd', 'e', 'f', and 'prob'
        """
        self.__identity = identity  # Identity assignment
        self.__iterations = iterations  # Number of fern iterations
        self.__non_fern_points = mask_points  # Number of non-fern points
        self.__min_distance = min_distance  # Minimum distance to avoid overlap
        self.__transformations = transformations if transformations else self.__default_transformations()
        self.__cumulative_probs = self.__compute_cumulative_probabilities()
        self.__datapoints: np.ndarray = np.empty((self.__iterations, 2), float)  # Fern datapoints
        self.__non_fern_datapoints: np.ndarray = np.empty((0, 2), float)  # Non-fern datapoints

    @staticmethod
    def __default_transformations() -> List[Dict[str, float]]:
        """
        Defines the default transformation functions and their probabilities for the Barnsley Fern.

        :return: List of transformation dictionaries
        """
        return [
            {"a": 0.0,   "b": 0.0,   "c": 0.0,   "d": 0.16,  "e": 0.0,  "f": 0.0,   "prob": 0.01},
            {"a": 0.85,  "b": 0.04,  "c": -0.04, "d": 0.85,  "e": 0.0,  "f": 1.6,   "prob": 0.85},
            {"a": 0.2,   "b": -0.26, "c": 0.23,  "d": 0.22,  "e": 0.0,  "f": 1.6,   "prob": 0.07},
            {"a": -0.15, "b": 0.28,  "c": 0.26,  "d": 0.24,  "e": 0.0,  "f": 0.44,  "prob": 0.07},
        ]

    def __compute_cumulative_probabilities(self) -> List[float]:
        """
        Computes the cumulative probabilities for the transformation selection.

        :return: List of cumulative probabilities
        """
        cumulative_probs = []
        cumulative = 0.0
        for trans in self.__transformations:
            cumulative += trans["prob"]
            cumulative_probs.append(cumulative)
        # Ensure that the last cumulative probability is exactly 1.0
        cumulative_probs[-1] = 1.0
        return cumulative_probs

    # Getter methods

    def get_identity(self) -> int:
        """
        :return: Identity of the Barnsley Fern
        """
        return self.__identity

    def get_datapoints(self) -> np.ndarray:
        """
        :return: Generated fern datapoints as a 2D NumPy array
        """
        return self.__datapoints

    def get_non_fern_datapoints(self) -> np.ndarray:
        """
        :return: Generated non-fern datapoints as a 2D NumPy array
        """
        return self.__non_fern_datapoints

    # Getter methods end

    def generate(self, n:int) -> np.ndarray:
        """
        Generates the Barnsley Fern using the Chaos Game method.

        :return: 2D NumPy array of generated fern points (x, y)
        """
        x, y = 0.0, 0.0  # Initialize starting point

        for i in range(self.__iterations):
            r = random.random()
            # Select transformation based on cumulative probabilities
            for j, trans in enumerate(self.__transformations):
                if r <= self.__cumulative_probs[j]:
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
        """
        Generates random non-fern datapoints within the fern's bounding box,
        ensuring they do not overlap with any fern datapoints, and stores them in Mask.npy.
        """
        x_min, x_max, y_min, y_max = self.get_bounding_box()
        fern_points = self.get_datapoints()

        # Build a KD-Tree for efficient nearest neighbor search
        tree = cKDTree(fern_points)

        # Initialize list to hold valid non-fern points
        valid_non_fern = []

        # Define batch size for generating points
        batch_size = 100000  # Adjust based on memory and performance considerations

        while len(valid_non_fern) < self.__non_fern_points:
            remaining = self.__non_fern_points - len(valid_non_fern)
            current_batch_size = min(batch_size, remaining * 2)  # Generate extra to account for rejections

            # Generate random points
            rand_x = np.random.uniform(x_min, x_max, current_batch_size)
            rand_y = np.random.uniform(y_min, y_max, current_batch_size)
            candidates = np.vstack((rand_x, rand_y)).T

            # Query the KD-Tree for points within min_distance
            indices = tree.query_ball_point(candidates, r=self.__min_distance)

            # Select points where no fern point is within min_distance
            mask = [len(idx) == 0 for idx in indices]
            valid_candidates = candidates[mask]

            # Add valid candidates to the list
            for point in valid_candidates:
                if len(valid_non_fern) < self.__non_fern_points:
                    valid_non_fern.append(point)
                else:
                    break

            # Continue until required number of non-fern points are generated

        self.__non_fern_datapoints = np.array(valid_non_fern)
        np.save("data_structures/Mask.npy", self.__non_fern_datapoints)

        labels = np.full((len(self.__datapoints), 1), self.__identity)
        self.__datapoints = np.concatenate((self.__datapoints, labels), axis=1)

        return self.__datapoints

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Computes the bounding box of the fern datapoints.

        :return: Tuple containing (x_min, x_max, y_min, y_max)
        """
        if self.__datapoints.size == 0:
            raise ValueError("Datapoints not generated. Call generate_set() first.")
        x_min, x_max = self.__datapoints[:, 0].min(), self.__datapoints[:, 0].max()
        y_min, y_max = self.__datapoints[:, 1].min(), self.__datapoints[:, 1].max()
        return x_min, x_max, y_min, y_max
