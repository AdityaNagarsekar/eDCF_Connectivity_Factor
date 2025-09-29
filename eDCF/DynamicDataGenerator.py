# import statements
from typing import List
import numpy as np


class DynamicDataGenerator:
    """
    This class generates dynamic data structure points according to specifications provided in the __init__() constructor.

    Output:
        Creates a Datapoints.npy file containing a 3D numpy array arr[i, j, k], where:
        
        i : int
            Structure number (index of the geometric structure in data_objects)
        j : int
            Point number within each structure (ranges from 0 to n_points-1)
        k : int
            Coordinate values plus identity label, where:
            - k[:-1] contains the n-dimensional coordinates
            - k[-1] contains the point's identity/label

    Note:
        The array structure allows representation of multiple geometric shapes,
        each with multiple points, where each point has n-dimensional coordinates
        plus an identity value.
"""

    def __init__(self, data_objects: List, num_points: int | List[int]):
        """
        Initialize the DynamicDataGenerator with geometric structures and point distribution specifications.

        Parameters:
            data_objects : List
                List of geometric structure objects (e.g., Circle, Spiral).
                Each object must implement a generate(n_points) method.
                
            num_points : int | List[int]
                Number of points to generate per structure:
                - If int: Same number of points for all structures
                - If List[int]: Individual point count for each structure
                            (length must match data_objects)

        Example:
            >>> # Create geometric structures with required parameters
            >>> circle = Circle(identity=1, radius=1.0, center=(0.0, 0.0), noise_rate=1.0)
            >>> spiral = Spiral(identity=2, angle_start=0, angle_end=720, 
            ...                center=(0.0, 0.0), noise_rate=1.0)
            >>> 
            >>> # Initialize generator with structures
            >>> generator = DynamicDataGenerator(
            ...     data_objects=[circle, spiral],
            ...     num_points=[1000, 2000]  # 1000 points for Circle, 2000 for Spiral
            ... )

        Note:
            - The order of data_objects determines the structure indices in the
            output array (i dimension in Datapoints.npy)
            - Each geometric structure must have a unique positive identity value
            - Structures must be properly initialized with required parameters
        """

        self.data_objects: List = data_objects

        self.n_points: int | List[int] = num_points

    @staticmethod
    def linear_transform(data_points: List) -> None:
        """
        Perform global normalization to fit all data points within the range [0, 1] across all dimensions.
        
        Parameters:
            data_points : List[np.ndarray]
                List of numpy arrays containing the generated points for each structure.
                Each array should have shape (n_points, n_dimensions + 1).
                
                Shape details:
                    n_points: number of points in the structure
                    n_dimensions + 1: coordinates plus identity label
                
                Array structure:
                    Columns [0:n]: Coordinate values for each dimension
                    Column [n+1]: Identity label
        
        Note:
            - Normalization is performed globally across all structures
            - Uses the same scale factor for all dimensions to preserve shape
            - Identity values (last column) are preserved unchanged
            - Original data is not modified; operates on a copy
            - Global normalization ensures consistent scaling across all structures
        
        Example:
            >>> # Given points in different scales
            >>> points1 = structure1.generate(1000)  # Points in range [-100, 100]
            >>> points2 = structure2.generate(1000)  # Points in range [0, 50]
            >>> data_points = [points1, points2]
            >>> DynamicDataGenerator.linear_transform(data_points)  # Global normalization
            >>> # Now all points are in range [0, 1] while preserving relative positions
        """

        datapoints = np.vstack(data_points.copy())  # Combine all structures' points into a single array for global min/max calculation

        grid_dimension = len(datapoints[0]) - 1  # Number of spatial dimensions (excluding identity column)

        # Calculate global min and max for each spatial dimension
        grid_bounds = [np.min(datapoints[:, : grid_dimension], axis=0),
                       np.max(datapoints[:, : grid_dimension], axis=0)]

        # Transpose to [n_dimensions, 2] shape where each row is [min, max] for that dimension
        grid_bounds = np.asarray(grid_bounds).transpose()

        # Find the largest range across all dimensions to maintain aspect ratio
        range_max = float("-inf")

        for i in range(len(grid_bounds)):
            if (grid_bounds[i, 1] - grid_bounds[i, 0]) > range_max:
                range_max = grid_bounds[i, 1] - grid_bounds[i, 0]

        # Normalize each dimension using the global range_max to preserve relative scales

        # ----------

        for i, bounds in enumerate(grid_bounds):  # parsing through dimensions/axes.

            for data in data_points:  # parsing through each data structure.

                data[:, i] = (data[:, i] - bounds[0]) / range_max  # conversion to 0-1 normal linear transformation.

                # NOTE: We could make it so that we have the data between the desired range of values by multiplying the RHS by (b - a) and adding 'a' thereafter for the range [b, a]

                # data[:, i] = data[:, i] * (b - a) + a

                # The above line of code can be uncommented for conversion to required linear transformation. Also take a and b as parameters.

        # ----------

    def generate_data(self, linear_transform: bool = False) -> None:
        """
        Generate and save data points from all geometric structures to 'Datapoints.npy'.
        
        Parameters:
            linear_transform : bool, default=False
                Whether to normalize the generated points to [0, 1] range.
                When True, applies global normalization across all structures.
        
        Output:
            Creates 'Datapoints.npy' file containing a 3D numpy array with shape:
            [n_structures, points_per_structure, n_dimensions + 1]
        
            The array contains:
                - One subarray per geometric structure
                - Points for each structure as specified by n_points
                - Coordinate values plus identity label for each point
        
        Note:
            - Number of points per structure is determined by self.n_points
            - If self.n_points is int: same number for all structures
            - If self.n_points is List[int]: individual counts per structure
            - Identity values are preserved during normalization
            - File is saved with pickle compatibility enabled
        
        Example:
            >>> generator = DynamicDataGenerator([circle, spiral], [1000, 2000])
            >>> generator.generate_data(linear_transform=True)  # Normalized output
            >>> # Creates Datapoints.npy with normalized points
        """

        data_points: List = []  # Initialize empty list to store generated points from all structures

        if isinstance(self.n_points, int):
            # Use same number of points for all structures
            for obj in self.data_objects:

                points = obj.generate(self.n_points)  # Generate points using structure's method
                data_points.append(np.asarray(points))  # Convert to numpy array for efficient processing

        else:
            # Use individual point counts for each structure
            for i, obj in enumerate(self.data_objects):

                points = obj.generate(self.n_points[i])  # Generate using structure-specific point count
                data_points.append(np.asarray(points))   # Convert to numpy array for efficient processing

        if linear_transform:  # Optional normalization if linear_transform is True
            DynamicDataGenerator.linear_transform(data_points)

        np.save("Datapoints", np.array(data_points, dtype=object), allow_pickle=True)  # Save to Datapoints.npy with pickle compatibility

        return None
