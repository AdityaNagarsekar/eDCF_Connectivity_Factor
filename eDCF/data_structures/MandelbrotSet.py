# Import statements
from typing import Dict, Tuple
import numpy as np
from scipy.spatial import cKDTree


class MandelbrotSet:
    """
    Generates Mandelbrot Set points and stores non-Mandelbrot points in Mask.npy.
    """

    def __init__(
        self,
        identity: int,
        iterations: int = 1000000,
        mask_points: int = 1000000,
        min_distance: float = 0.01,
        parameters: Dict[str, float] = None,
        power: int = 2
    ):
        """
        Constructor for MandelbrotSet

        :param identity: Unique identifier for the Mandelbrot Set instance (identity > 0)
        :param iterations: Number of points to generate for the Mandelbrot Set
        :param mask_points: Number of non-Mandelbrot points to generate
        :param min_distance: Minimum distance from any Mandelbrot point to consider a non-Mandelbrot point valid
        :param parameters: Dictionary of parameters for Mandelbrot set generation
        """
        self.__power = power
        self.__identity = identity  # Identity assignment
        self.__iterations = iterations  # Number of Mandelbrot iterations
        self.__non_mandelbrot_points = mask_points  # Number of non-Mandelbrot points
        self.__min_distance = min_distance  # Minimum distance to avoid overlap
        self.__parameters = parameters if parameters else self.__default_parameters()
        self.__datapoints: np.ndarray = np.empty((self.__iterations, 2), float)  # Mandelbrot datapoints
        self.__non_mandelbrot_datapoints: np.ndarray = np.empty((0, 2), float)  # Non-Mandelbrot datapoints

    @staticmethod
    def __default_parameters() -> Dict[str, float]:
        """
        Defines the default parameters for the Mandelbrot Set.

        :return: Dictionary of parameters
        """
        return {
            "x_min": -2.0,
            "x_max": 1.0,
            "y_min": -1.5,
            "y_max": 1.5,
            "max_iter": 100
        }

    # Getter methods

    def get_identity(self) -> int:
        """
        :return: Identity of the Mandelbrot Set
        """
        return self.__identity

    def get_datapoints(self) -> np.ndarray:
        """
        :return: Generated Mandelbrot datapoints as a 2D NumPy array
        """
        return self.__datapoints

    def get_non_mandelbrot_datapoints(self) -> np.ndarray:
        """
        :return: Generated non-Mandelbrot datapoints as a 2D NumPy array
        """
        return self.__non_mandelbrot_datapoints

    # Getter methods end

    def generate(self, n: int) -> np.ndarray:
        params = self.__parameters
        x_min = params["x_min"]
        x_max = params["x_max"]
        y_min = params["y_min"]
        y_max = params["y_max"]
        max_iter = params["max_iter"]

        mandelbrot_points = []
        num_points_generated = 0

        while num_points_generated < self.__iterations:
            batch_size = 10000  # Adjust as needed
            x = np.random.uniform(x_min, x_max, batch_size)
            y = np.random.uniform(y_min, y_max, batch_size)
            C = x + 1j * y
            Z = np.zeros_like(C)
            mask = np.ones(batch_size, dtype=bool)

            for _ in range(int(max_iter)):
                Z[mask] = Z[mask] ** self.__power + C[mask]
                mask[mask] = np.abs(Z[mask]) < 2

            bounded_points = C[mask]
            mandelbrot_points.extend(bounded_points)
            num_points_generated += len(bounded_points)

        mandelbrot_points = np.array(mandelbrot_points[:self.__iterations])
        self.__datapoints = np.column_stack((mandelbrot_points.real, mandelbrot_points.imag))

        # Generate non-Mandelbrot data as before
        mandelbrot_points_data = self.get_datapoints()
        tree = cKDTree(mandelbrot_points_data)

        valid_non_mandelbrot = []
        batch_size = 100000

        while len(valid_non_mandelbrot) < self.__non_mandelbrot_points:
            remaining = self.__non_mandelbrot_points - len(valid_non_mandelbrot)
            current_batch_size = min(batch_size, remaining * 2)

            rand_x = np.random.uniform(x_min, x_max, current_batch_size)
            rand_y = np.random.uniform(y_min, y_max, current_batch_size)
            candidates = np.vstack((rand_x, rand_y)).T

            distances, _ = tree.query(candidates, k=1)
            mask = distances >= self.__min_distance

            valid_candidates = candidates[mask]
            valid_non_mandelbrot.extend(valid_candidates[:remaining])

        self.__non_mandelbrot_datapoints = np.array(valid_non_mandelbrot[:self.__non_mandelbrot_points])
        np.save("data_structures/Mask.npy", self.__non_mandelbrot_datapoints)

        labels = np.full((len(self.__datapoints), 1), self.__identity)
        self.__datapoints = np.concatenate((self.__datapoints, labels), axis=1)

        return self.__datapoints

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Computes the bounding box of the Mandelbrot datapoints.

        :return: Tuple containing (x_min, x_max, y_min, y_max)
        """
        if self.__datapoints.size == 0:
            raise ValueError("Datapoints not generated. Call generate() first.")
        x_min, x_max = self.__datapoints[:, 0].min(), self.__datapoints[:, 0].max()
        y_min, y_max = self.__datapoints[:, 1].min(), self.__datapoints[:, 1].max()
        return x_min, x_max, y_min, y_max
