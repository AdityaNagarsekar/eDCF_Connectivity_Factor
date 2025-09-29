from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


class JuliaSet:
    """
    Generates Julia Set boundary points for visualization.
    """

    def __init__(
            self,
            identity: int,
            iterations: int = 100000,
            boundary_threshold: float = 0.01,
            c: complex = -0.122561 + 0.744861j,
            max_iter: int = 100,
            escape_radius: float = 2.0
    ):
        """
        Constructor for JuliaSet focused on boundary visualization

        :param identity: Unique identifier for the Julia Set instance (identity > 0)
        :param iterations: Number of points to generate for the Julia Set boundary
        :param boundary_threshold: Threshold to determine if a point is on the boundary
        :param c: Complex parameter for the Julia Set formula Z = Z^2 + c
        :param max_iter: Maximum iterations for escape time algorithm
        :param escape_radius: Escape radius for determining boundedness
        """
        self.__identity = identity
        self.__iterations = iterations
        self.__boundary_threshold = boundary_threshold
        self.__c = c
        self.__max_iter = max_iter
        self.__escape_radius = escape_radius
        self.__datapoints = np.empty((0, 3), float)  # [x, y, label]

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

    def is_boundary_point(self, z: complex) -> bool:
        """
        Determines if a point is on the boundary of the Julia set by checking
        if it has neighbors with different escape behaviors.

        :param z: Complex point to check
        :return: True if point is on the boundary, False otherwise
        """
        # Check the behavior of the center point
        center_escapes = self.escape_time(z) < self.__max_iter

        # Check neighbors in a small radius
        delta = self.__boundary_threshold
        neighbors = [
            z + delta,
            z - delta,
            z + delta * 1j,
            z - delta * 1j
        ]

        # If any neighbor has different escape behavior, this is a boundary point
        for neighbor in neighbors:
            neighbor_escapes = self.escape_time(neighbor) < self.__max_iter
            if neighbor_escapes != center_escapes:
                return True

        return False

    def escape_time(self, z: complex) -> int:
        """
        Calculate the escape time for a given point in the complex plane.

        :param z: Complex point to check
        :return: Number of iterations before escape, or max_iter if bounded
        """
        c = self.__c
        n = 0
        while abs(z) < self.__escape_radius and n < self.__max_iter:
            z = z * z + c
            n += 1
        return n

    def generate(self, n: int) -> np.ndarray:
        """
        Generates the Julia Set boundary points.

        :return: 2D NumPy array of generated Julia Set boundary points (x, y) with labels
        """
        # Define the region of interest
        x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0

        # Store boundary points
        boundary_points = []

        # Grid-based search for boundary points is more efficient than random sampling
        grid_size = int(np.sqrt(self.__iterations * 10))  # Oversample to ensure we get enough points
        x_vals = np.linspace(x_min, x_max, grid_size)
        y_vals = np.linspace(y_min, y_max, grid_size)

        count = 0

        # Systematically check grid points for boundary status
        for x in x_vals:
            for y in y_vals:
                z = complex(x, y)
                if self.is_boundary_point(z):
                    boundary_points.append((x, y))
                    count += 1
                    if count >= self.__iterations:
                        break
            if count >= self.__iterations:
                break

        # If we didn't find enough boundary points with the grid approach,
        # supplement with random sampling
        while len(boundary_points) < self.__iterations:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = complex(x, y)
            if self.is_boundary_point(z):
                boundary_points.append((x, y))

        # Convert to numpy array and add labels
        boundary_array = np.array(boundary_points[:self.__iterations])
        labels = np.full((boundary_array.shape[0], 1), self.__identity)
        self.__datapoints = np.hstack((boundary_array, labels))

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

    def visualize(self, filename=None):
        """
        Visualize the Julia set boundary

        :param filename: If provided, save the image to this file
        """
        if self.__datapoints.size == 0:
            raise ValueError("Datapoints not generated. Call generate() first.")

        plt.figure(figsize=(10, 10), facecolor='black')
        plt.scatter(
            self.__datapoints[:, 0],
            self.__datapoints[:, 1],
            s=0.05,
            color='white',
            alpha=0.8
        )
        plt.axis('equal')
        plt.title(f"Julia Set (c = {self.__c})", color='white', fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, facecolor='black')

        plt.show()