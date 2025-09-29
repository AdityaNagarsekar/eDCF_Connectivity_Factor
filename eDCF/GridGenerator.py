# import statements
import numpy as np
import multiprocessing
import psutil


class GridGenerator:
    """
    Generate an n-dimensional grid of evenly spaced points and label them using a trained algorithm.

    Parameters
    ----------
    algorithm : object
        Trained algorithm instance with `initialize_hyper` and `execute` methods.

    Attributes
    ----------
    __data_set : np.ndarray
        Loaded data points from 'Datapoints.npy'.
    __algorithm : object
        Algorithm used to assign labels to grid points.

    Methods
    -------
    algorithm_set() -> None
        Initialize algorithm hyperparameters.
    mini_grid_compute(grid_bounds: np.ndarray, spacing: float) -> np.ndarray
        Compute a sub-grid within given bounds.
    compute() -> None
        Generate the full grid and save to 'Grid_Points.npy'.

    Examples
    --------
    >>> from algorithms.MLP import MLP
    >>> mlp = MLP(...)
    >>> generator = GridGenerator(mlp)
    >>> generator.algorithm_set()
    >>> generator.compute()
    """

    def __init__(self, algorithm):
        """
        Initialize GridGenerator with a trained algorithm.

        Parameters
        ----------
        algorithm : object
            Trained algorithm instance with `initialize_hyper()` and `execute()` methods.

        Notes
        -----
        Requires 'Datapoints.npy' to exist in the current working directory.
        """

        # loading datapoints.
        self.__data_set = np.load("Datapoints.npy", allow_pickle=True)

        # assigning algorithm
        self.__algorithm = algorithm

    def algorithm_set(self) -> None:
        """
        Load and apply the algorithm's best hyperparameters.

        Workflow
        --------
        1. Call `self.__algorithm.initialize_hyper()` to load saved parameters and retrain the algorithm.

        Returns
        -------
        None
        """

        self.__algorithm.initialize_hyper()

        return None

    def mini_grid_compute(self, grid_bounds: np.ndarray, spacing: float) -> np.ndarray:
        """
        Compute a sub-grid within specified bounds and label each point using multiprocessing.

        Parameters
        ----------
        grid_bounds : np.ndarray, shape (n_dims, 2)
            Array of [lower_bound, upper_bound] pairs for each dimension.
        spacing : float
            Distance between adjacent points along each axis.

        Returns
        -------
        np.ndarray, shape (n_points, n_dims + 1)
            Grid points with predicted labels in the last column.
        """

        grid_axes = []  # to store the points of the axes which are evenly spaced.

        for i_l in range(len(grid_bounds)):  # iterating through dimensions

            lower_bound, upper_bound = grid_bounds[i_l]

            num_points = (grid_bounds[i_l, 1] - grid_bounds[i_l, 0]) / spacing + 1  # checking the divisions between lower bound and upper bound (both inclusive technically)
            num_points = np.round(num_points)  # rounding off because of computational inaccuracies otherwise (0.1 + 0.1 is not equal to 0.2 strictly)
            num_points = int(num_points)  # converting num_points to integer value

            axis_points = []  # to store the points in a particular axis

            for i in range(num_points):  # iterating through the number of points between lower bound and upper bound
                axis_points.append(lower_bound + i * spacing)  # appending the coordinates to list

            axis_points = np.asarray(axis_points)  # conversion to numpy array from list
            grid_axes.append(axis_points)  # adding an axis to the list containing axes

        grids = np.meshgrid(*grid_axes, indexing='ij')  # using mesh grid to make grids from the given axes.

        grid_points = np.stack(grids, axis=-1)  # getting grid points

        grid_points = grid_points.reshape(-1, len(grid_bounds))  # reshaping the grid_points to the format we want

        labels = np.full((grid_points.shape[0], 1), 0.0)  # constructing labels to add

        grid_points = np.hstack((grid_points, labels))  # adding initialized labels of 0.0 to grid_points

        for point in grid_points:  # iterating through all grid_points

            assigned_label: float = self.__algorithm.execute(point)  # assigning a value to these grid_points based on classifier

            point[-1] = assigned_label  # changing initialized label to assigned_label.

        return grid_points

    @staticmethod
    def get_available_cores(threshold: int = 50) -> int:
        """
        Get the number of CPU cores available for computation based on current usage.

        Parameters
        ----------
        threshold : int, optional
            CPU usage percentage threshold at which a core is considered busy (default: 50).

        Returns
        -------
        int
            Number of CPU cores available for use.
        """

        num_cores = multiprocessing.cpu_count()
        current_usage = psutil.cpu_percent(interval=1)

        # If current CPU usage is above the threshold, reduce the number of available cores
        available_cores = num_cores
        if current_usage > threshold:
            # Reduce the number of available cores based on the usage
            available_cores = max(1, num_cores - int(current_usage / 100 * num_cores))

        return available_cores

    def super_loop(self, dimension: int, max_dim: int, record_i: np.ndarray, max_num: int, grid_params: np.ndarray, final_vals: list):
        """
        Recursively compute mini-grid boundaries for all combinations of partitions.

        Parameters
        ----------
        dimension : int
            Current recursion depth (start at `max_dim`).
        max_dim : int
            Total number of dimensions in the grid.
        record_i : np.ndarray, shape (max_dim,)
            Index vector indicating the current partition indices along each axis.
        max_num : int
            Number of divisions per axis in the full grid partitioning.
        grid_params : np.ndarray, shape (max_num, max_dim, 2)
            Boundary pairs ([lower, upper]) for each partition and dimension.
        final_vals : list
            Accumulator list of boundary combinations from previous recursion steps.

        Returns
        -------
        final_vals : list
            Updated list of all boundary combinations (unique after post-processing).
        vals : np.ndarray, shape (n_vals, max_dim * 2)
            Array of boundary pairs for current recursion level.
        """

        vals = []  # initializing an empty list for holding values

        for i_l in range(max_num):  # iterating through the divisions/partition number

            vals_basic = []  # vals_basic is initialized to be an empty list which will actually hold the values of lower and upper bounds for each dimension.

            record_i[max_dim - dimension] += 1  # increasing the current dimensions record (we want the dimension depth to be opposite of index in record {not necessary to do this particular way})

            if (dimension - 1) != 0:  # if the next dimension is not zero/current dimension is not 1

                final_vals, vals = self.super_loop(dimension - 1, max_dim, np.copy(record_i), max_num, grid_params,
                                              final_vals)
                # Recursive call with copy of record being passed and not the reference. Note that sensing a copy instead of reference is crucial here.

                if vals.size != 0:  # if vals that is returned is not zero then only append it to final_vals. Basically a safety check.
                    final_vals.append(vals)

            if dimension == 1:  # In case the dimension is 1

                for j_l in range(max_dim):  # start adding bounds to vals_basic over dimensions
                    vals_basic.append(grid_params[record_i[j_l], j_l])

                vals.append(vals_basic)  # add the max_dim dimensional block formed in vals_basic over max_dim to vals

        vals = np.asarray(vals)  # convert vals to a numpy array for addition.

        return final_vals, vals

    def compute(self) -> None:
        """
        Generate and save the complete labelled grid of points.

        Workflow
        --------
        1. Retrieve available CPU cores via `get_available_cores()`.
        2. Load full grid bounds from 'Grid_Bounds.npy' and metadata from 'Grid_Info.npy'.
        3. Partition bounds into mini-grid segments.
        4. Use a multiprocessing pool to compute each mini-grid with `mini_grid_compute`.
        5. Stack all mini-grids and save the result array to 'Grid_Points.npy'.

        Returns
        -------
        None

        Notes
        -----
        Output file: 'Grid_Points.npy'
        Output shape: (n_blocks, n_points_per_block, n_dims + 1)
        Each entry `grid_points[i, j, k]` corresponds to block i, point j, coordinate k.
        """

        # Get the number of available CPU cores based on current usage
        num_workers = GridGenerator.get_available_cores(threshold=50)

        grid_bounds_full = np.load("Grid_Bounds.npy", allow_pickle=True)  # loading the 'complete grid' bounds.

        info = np.load("Grid_Info.npy")  # loading grid info.
        spacing_points: float = info[0]  # getting spacing for points

        n = (((grid_bounds_full[:, 1] - grid_bounds_full[:, 0]) // spacing_points) + 1).astype(int)  # number of points between full bounds in each dimension

        n = (n // info[2]).astype(int)  # number of points assigned to each mini grid.

        g_space: np.ndarray[float] = n * spacing_points  # Each mini grid should have these spacings in each of the dimensions

        lowers = []  # lower bound list to store the lower bounds
        uppers = []  # upper bound list to store the upper bounds

        for i in range(len(grid_bounds_full)):  # parsing through dimensions

            lowers_temp = []  # temporary holding of lower bound
            uppers_temp = []  # temporary list for holding upper bound

            for j in range(int(info[2])):  # parsing through mini grids to generate lower and upper bounds for this specific dimension

                lowers_temp.append(np.asarray(grid_bounds_full[i, 0] + j * g_space[i]))
                uppers_temp.append(np.asarray(grid_bounds_full[i, 0] + (j + 1) * g_space[i]))

            lowers.append(np.asarray(lowers_temp))  # appending the lower bounds for each mini grid in current dimension
            uppers.append(np.asarray(uppers_temp))  # appending the upper bounds for each mini grid in current dimension

        lowers = np.asarray(lowers)
        uppers = np.asarray(uppers)

        grid_bounds_params = []  # to store the grid_params in the required format in super_loop (please check doc string of that function to see what exactly is required)

        for i in range(int(info[2])):  # parsing through partition/grid numbers
            combined = []  # combining list initializing

            for j in range(int(info[3])):
                combined.append([lowers[j][i], uppers[j][i]])  # essentially creates a mini grid of index (j, j, j, ... i terms) where this format's explanation is given in super_loop method.

            # Append the combined bounds for this coordinate
            grid_bounds_params.append(np.asarray(combined))

        grid_bounds_params = np.asarray(grid_bounds_params)

        final_grid_bounds_param = []  # getting ready to pass the reference of an empty list which will eventually be filled to have all the mini grid bounds in the proper format.

        # this recursion is explained in the super_loop function please refer to that...
        final_grid_bounds_param = np.unique(np.asarray(self.super_loop(dimension=int(info[3]), max_dim=int(info[3]), record_i=np.full((int(info[3])), -1), max_num=int(info[2]), grid_params=grid_bounds_params, final_vals=final_grid_bounds_param)[0]), axis=0)

        final_grid_bounds_param = final_grid_bounds_param.reshape(-1, int(info[3]), 2)  # should do this to ensure proper format as required.

        np.save("Divided_Grid_Bounds", final_grid_bounds_param)  # saving the divided grid bounds if ever required.

        spacing_points_param = np.full((len(final_grid_bounds_param)), spacing_points)  # spacing_points is now made into an array for distributions to various workers in multiprocessing.Pool

        # preparation for multiprocessing

        # ----------

        final_grid_bounds_param_list = []
        spacing_points_param_list = []

        for i in range(len(final_grid_bounds_param)):
            final_grid_bounds_param_list.append(final_grid_bounds_param[i])
            spacing_points_param_list.append(spacing_points_param[i])

        params = list(zip(final_grid_bounds_param, spacing_points_param))

        # Create a Pool with the calculated number of available CPU cores
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use pool.starmap to unpack the parameter tuples
            grid_points = pool.starmap(self.mini_grid_compute, params)

        # ----------

        grid_points = np.asarray(grid_points)

        np.save("Grid_Points", grid_points)  # saving the grid points that have been generated.

        return None
