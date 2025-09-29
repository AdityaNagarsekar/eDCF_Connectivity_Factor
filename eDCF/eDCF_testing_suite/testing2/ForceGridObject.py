import numpy as np
import multiprocessing
from typing import Tuple
import psutil
from joblib import Parallel, delayed 


class ForceGridObject:
    """
    ForceGridObject enforces grid alignment on irregular data via box-counting.

    Transforms non-uniform point clouds into evenly spaced grids and
    generates hatch points based on data presence.

    Attributes
    ----------
    __data_set : numpy.ndarray
        Raw data points loaded by identity.
    n : int
        Number of points in the data_set.

    Methods
    -------
    count_decimal_places(number, max_to_consider_trailing=8) -> int
    process(num, spacing, precision) -> float
    mini_grid_compute(grid_bounds, spacing) -> (np.ndarray, np.ndarray)
    get_available_cores(threshold=50) -> int
    super_loop(dimension, max_dim, record_i, max_num, grid_params, final_vals)
    compute(direct_conversion_ctrl=False, spacing_ctrl=False, changing_factor=5) -> bool
    grid_points_finder(num_workers, grid_bounds_full, info, spacing_points, direct_conversion_ctrl=False)
    spacing_allocator(num_workers, grid_bounds_full, info, changing_factor=5, direct_conversion_ctrl=False)
    direct_gridder(data_set, grid_bounds_full, info, spacing) -> set
    """

    def __init__(self, identity: int):
        """
        Initialize ForceGridObject with data identity.

        Parameters
        ----------
        identity : int
            1-based index to select subset from 'Datapoints.npy'.

        Raises
        ------
        FileNotFoundError
            If 'Datapoints.npy' cannot be loaded.
        """
        
        # loading datapoints.
        self.__data_set = np.load("Datapoints.npy", allow_pickle=True)[identity - 1]
        self.n: int = len(self.__data_set)
        self.__identity = identity
        self.__limit = 14

    @staticmethod
    def count_decimal_places(number, max_to_consider_trailing: int = 8):
        """
        Count decimal places in a number string, handling scientific notation.

        Parameters
        ----------
        number : float or int
            Value to inspect.
        max_to_consider_trailing : int, optional
            Max trailing zeros to consider (default: 8).

        Returns
        -------
        int
            Total decimal places.

        Notes
        -----
        Parses exponent notation and counts non-zero digits.
        """
        index = str(number).find('e')

        decimals = str(number).split(".")[1]
        count: int = 0
        i: int = 0
        decimal_count: int = 0

        if index == -1:
            index = len(decimals)
        else:
            decimal_count = int(str(number)[index + 2:])
            index -= 2

        while count < max_to_consider_trailing and i < index:
            if decimals[i] == '0':
                count += 1
            else:
                decimal_count += count + 1
                count = 0
            i += 1

        return decimal_count + 1

    @staticmethod
    def process(num: float, spacing: float, precision: int) -> float:
        """
        Align numeric value to a specified grid spacing.

        Parameters
        ----------
        num : float
            Original value.
        spacing : float
            Grid spacing distance.
        precision : int
            Decimal precision for scaling.

        Returns
        -------
        float
            Rounded value to nearest grid cell center.

        Notes
        -----
        Converts to integer grid units, floors to spacing multiple,
        then rescales and offsets by half spacing.
        """
        num = num * (10 ** precision)

        num_int: int = int(num)
        spacing_int: int = int(spacing * (10 ** precision))

        num_int = num_int - num_int % spacing_int

        num = num_int * (10 ** (-precision))

        num += spacing / 2

        return round(num, precision + 1)

    def mini_grid_compute(self, grid_bounds, spacing) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sub-grid and hatch points within given bounds.

        Parameters
        ----------
        grid_bounds : numpy.ndarray, shape (n_dims, 2)
            Lower and upper bounds for each axis.
        spacing : float
            Grid spacing distance.

        Returns
        -------
        grid_points : numpy.ndarray, shape (m, n_dims+1)
            Grid cell coordinates with labels in last column.
        hatch : numpy.ndarray, shape (k, n_dims+1)
            Points to hatch based on data presence.

        Notes
        -----
        Filters data_set points within each grid cell and labels boxes by identity.
        """
        grid_axes = []

        for i_l in range(len(grid_bounds)):
            lower_bound, upper_bound = grid_bounds[i_l]

            num_points = (grid_bounds[i_l, 1] - grid_bounds[i_l, 0]) / spacing + 1
            num_points = round(num_points)
            num_points = int(num_points)

            axis_points = []

            for i in range(num_points):
                axis_points.append(lower_bound + i * spacing)

            axis_points = np.asarray(axis_points)
            grid_axes.append(axis_points)

        grids = np.meshgrid(*grid_axes, indexing='ij')

        grid_points = np.stack(grids, axis=-1)

        grid_points = grid_points.reshape(-1, len(grid_bounds))

        data_set = []

        # Taking only points are there in the particular mini grid for comparisons

        # ----------

        for point in self.__data_set:
            if ((grid_bounds[:, 0] <= point[:-1]) & (point[:-1] <= grid_bounds[:, 1])).all():
                data_set.append(point)

        data_set = np.asarray(data_set)

        # ----------

        labels = np.full((grid_points.shape[0], 1), 0.0)

        grid_points = np.hstack((grid_points, labels))  # taking grid_points labelled to 0.0 by default

        hatch = []  # initializing an empty hatch list to fill

        if len(data_set) != 0:  # if there are points in the data_set then only do this

            for box in grid_points:  # for every box point in grid_points

                for point in data_set:  # iterating through every point in data_set variable for this box

                    if (box[:-1] - (spacing / 2) < point[:-1]).all() and (point[:-1] < box[:-1] + (spacing / 2)).all():  # we see if any point lies in this box.
                        box[-1] = self.__identity  # identified the box to have the identity of the data_set
                        break

                box_copy = box.copy()
                box_copy[-1] = self.__identity
                hatch.append(box_copy) # regardless of whether this particular box has the point or not since the mini grid consists of a point we take the box as a hatch point.

        return grid_points, np.asarray(hatch)

    @staticmethod
    def get_available_cores(threshold: int = 50) -> int:
        """
        Determine available CPU cores based on system usage.

        Parameters
        ----------
        threshold : int, optional
            CPU usage percent threshold (default: 50).

        Returns
        -------
        int
            Number of CPU cores below the usage threshold.

        Notes
        -----
        Uses psutil.cpu_percent over a 1-second interval.
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
        Recursively generate boundary combinations for grid partitions.

        Parameters
        ----------
        dimension : int
            Remaining recursion depth.
        max_dim : int
            Total number of grid dimensions.
        record_i : numpy.ndarray, shape (max_dim,)
            Index tracker for combination state.
        max_num : int
            Number of divisions per axis.
        grid_params : numpy.ndarray, shape (max_num, max_dim, 2)
            Division bounds per axis.
        final_vals : list
            Accumulator for boundary combinations.

        Returns
        -------
        final_vals : list
            Updated boundary combinations.
        vals : numpy.ndarray
            New combinations at this level.

        Notes
        -----
        Duplicate combinations may occur; apply np.unique afterwards.
        """
        vals = []

        for i_l in range(max_num):

            vals_basic = []

            record_i[max_dim - dimension] += 1

            if (dimension - 1) != 0:

                final_vals, vals = self.super_loop(dimension - 1, max_dim, np.copy(record_i), max_num, grid_params,
                                              final_vals)

                if vals.size != 0:
                    final_vals.append(vals)

            if dimension == 1:

                for j_l in range(max_dim):
                    vals_basic.append(grid_params[record_i[j_l], j_l])

                vals.append(vals_basic)

        vals = np.asarray(vals)

        return final_vals, vals

    def compute(self, direct_conversion_ctrl: bool = False, spacing_ctrl: bool = False, changing_factor: int = 5) -> bool:
        """
        Execute grid forcing workflow and save outputs.

        Parameters
        ----------
        direct_conversion_ctrl : bool, optional
            Skip hatching when True (default: False).
        spacing_ctrl : bool, optional
            Enable dynamic spacing allocation when True (default: False).
        changing_factor : int, optional
            Factor to adjust spacing during allocation (default: 5).

        Returns
        -------
        bool
            False if required files missing; True on success.

        Notes
        -----
        Saves hatch to 'Hatch_Object_{id}', grid points, and percent metrics.
        """

        # Get the number of available CPU cores based on current usage
        num_workers = ForceGridObject.get_available_cores(threshold=50)

        percent = 0.0

        try:
            grid_bounds_full = np.load("Grid_Bounds.npy", allow_pickle=True)
            info = np.load("Grid_Info.npy")
        except FileNotFoundError as file_error:
            print(file_error)
            print("Recalculation of Grid Parameters Required.")
            return False

        if not spacing_ctrl:
            spacing_points = info[0]

            if not direct_conversion_ctrl:
                grid_points, hatch = self.grid_points_finder(num_workers, grid_bounds_full, info, spacing_points)
                np.save(f"Hatch_Object_{self.__identity}", hatch)
            else:
                grid_points = self.grid_points_finder(num_workers, grid_bounds_full, info, spacing_points, direct_conversion_ctrl)

        else:

            if not direct_conversion_ctrl:
                hatch, grid_points, spacing_points, percent = self.spacing_allocator(num_workers, grid_bounds_full, info, changing_factor)
                np.save(f"Hatch_Object_{self.__identity}", hatch)

            else:

                grid_points, spacing_points, percent = self.spacing_allocator(num_workers, grid_bounds_full, info, changing_factor, direct_conversion_ctrl)

        info_list = list(info)
        info_list.append(spacing_points)
        np.save("Grid_Info.npy", np.asarray(info_list))
        np.save(f"Grid_Points_Object_{self.__identity}", grid_points)
        np.save(f"Percent_{self.__identity}", np.array([percent]))

        return True

    def grid_points_finder(self, num_workers: int, grid_bounds_full: np.ndarray, info: np.ndarray, spacing_points: float, direct_conversion_ctrl: bool = False):
        """
        Partition global bounds into subgrids and compute points.

        Parameters
        ----------
        num_workers : int
            Number of parallel workers.
        grid_bounds_full : numpy.ndarray, shape (n_dims, 2)
            Global axis bounds.
        info : numpy.ndarray
            Grid configuration [spacing, buffer, divide, dimension].
        spacing_points : float
            Grid spacing distance.
        direct_conversion_ctrl : bool, optional
            Skip hatching when True (default: False).

        Returns
        -------
        grid_points : numpy.ndarray or list
            Computed grid points (and hatch info if applicable).
        """
        if not direct_conversion_ctrl:
            n = (((grid_bounds_full[:, 1] - grid_bounds_full[:, 0]) // spacing_points) + 1).astype(int)

            n = (n // info[2]).astype(int)

            g_space = n * spacing_points

            # Create the grid for each axis using grid_bounds_full and g_space

            lowers = []
            uppers = []

            for i in range(len(grid_bounds_full)):

                lowers_temp = []
                uppers_temp = []

                for j in range(int(info[2])):
                    lowers_temp.append(np.asarray(grid_bounds_full[i, 0] + j * g_space[i]))
                    uppers_temp.append(np.asarray(grid_bounds_full[i, 0] + (j + 1) * g_space[i]))

                lowers.append(np.asarray(lowers_temp))
                uppers.append(np.asarray(uppers_temp))

            lowers = np.asarray(lowers)
            uppers = np.asarray(uppers)

            grid_bounds_params = []

            for i in range(int(info[2])):
                combined = []

                # For each dimension, combine the lower and upper bound for the i-th coordinate
                for j in range(int(info[3])):
                    combined.append([lowers[j][i], uppers[j][i]])

                # Append the combined bounds for this coordinate
                grid_bounds_params.append(np.asarray(combined))

            # Convert to numpy array if needed (optional)
            grid_bounds_params = np.asarray(grid_bounds_params)

            final_grid_bounds_param = []
            final_grid_bounds_param = np.unique(np.asarray(
                self.super_loop(dimension=int(info[3]), max_dim=int(info[3]), record_i=np.full((int(info[3])), -1),
                                max_num=int(info[2]), grid_params=grid_bounds_params, final_vals=final_grid_bounds_param)[
                    0]), axis=0)

            final_grid_bounds_param = final_grid_bounds_param.reshape(-1, int(info[3]), 2)

            np.save(f"Divided_Grid_Bounds_Object_{self.__identity}", final_grid_bounds_param)

            spacing_points_param = np.full((len(final_grid_bounds_param)), spacing_points)

            final_grid_bounds_param_list = []
            spacing_points_param_list = []
            for i in range(len(final_grid_bounds_param)):
                final_grid_bounds_param_list.append(final_grid_bounds_param[i])
                spacing_points_param_list.append(spacing_points_param[i])

            params = list(zip(final_grid_bounds_param, spacing_points_param))

            # Create a Pool with the calculated number of available CPU cores
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Use pool.starmap to unpack the parameter tuples
                points = pool.starmap(self.mini_grid_compute, params)

            grid_points = []
            hatch = []

            for i in range(len(points)):

                grid_points.append(points[i][0])
                if len(points[i][1]) != 0:
                    hatch.append(points[i][1])

            grid_points = np.asarray(grid_points)
            hatch = np.asarray(hatch)

            return grid_points, hatch

        else:

            if len(self.__data_set) > num_workers:
                chunk_size = int(np.ceil(len(self.__data_set) / num_workers))
                data_set_list = [
                    self.__data_set[i : i + chunk_size]
                    for i in range(0, len(self.__data_set), chunk_size)
                ]
            else:
                data_set_list = [self.__data_set]

            # broadcast shared args
            params = [
                (ds, grid_bounds_full, info, spacing_points) for ds in data_set_list
            ]

            # ---------------- PARALLEL (threads) ----------------
            def _worker(ds, gb, inf, sp):
                # self.direct_gridder is thread-safe
                return self.direct_gridder(ds, gb, inf, sp)

            grid_set_list = Parallel(
                n_jobs=num_workers, backend="threading", prefer="threads"
            )(delayed(_worker)(*p) for p in params)

            # ---------------- combine ----------------
            final_set: set = set().union(*grid_set_list)
            grid_points = np.array(list(final_set))
            return grid_points

    def spacing_allocator(self, num_workers: int, grid_bounds_full: np.ndarray, info: np.ndarray, changing_factor: int = 5, direct_conversion_ctrl: bool = False):
        """
        Allocate spacing to meet data coverage criteria.

        Parameters
        ----------
        num_workers : int
            Number of worker processes.
        grid_bounds_full : numpy.ndarray
            Global grid bounds.
        info : numpy.ndarray
            Grid parameters.
        changing_factor : int, optional
            Factor to adjust spacing (default: 5).
        direct_conversion_ctrl : bool, optional
            Skip hatching when True (default: False).

        Returns
        -------
        hatch_final : numpy.ndarray
            Selected hatch points.
        grid_points_final : numpy.ndarray
            Selected grid points.
        spacing_points : float
            Final spacing value used.
        percent : float
            Coverage percentage achieved.
        """
        spacing_allocation_details = np.load("Spacing_Allocation_Details.npy")
        percent = 0.0

        spacing_points = spacing_allocation_details[0] # Assigning start checker

        if not (0 <= spacing_allocation_details[2] <= 100 and 0 <= spacing_allocation_details[3] <= 100):
            raise ValueError("Percentage Value MUST be between 0 and 100 (both inclusive)")

        print("\nVVV Starting Tracking Percentage of the Total Set that Force Griding has reduced down to VVV")

        hatch_final = []
        grid_points_final = []

        while spacing_points >= spacing_allocation_details[1]: # Checking if spacing has crossed the final checker allowed

            hatch = np.asarray([])

            if not direct_conversion_ctrl:
                grid_points, hatch = self.grid_points_finder(num_workers, grid_bounds_full, info, spacing_points)
            else:
                grid_points = self.grid_points_finder(num_workers, grid_bounds_full, info, spacing_points, direct_conversion_ctrl)

            if not direct_conversion_ctrl:
                num_points: int = len(grid_points[grid_points[:, :, -1] == self.__identity])
            else:
                num_points: int = len(grid_points)

            percent = num_points / self.n * 100

            print(f"\nSpacing - {spacing_points} and Percentage - {percent}%")

            if spacing_allocation_details[2] <= percent < spacing_allocation_details[3]:

                hatch_final = hatch.copy()
                grid_points_final = grid_points.copy()
                break

            elif percent < spacing_allocation_details[2]:

                spacing_points /= changing_factor
                spacing_points = round(spacing_points, self.__limit)

            else:

                low: float = spacing_points
                high: float = spacing_points * changing_factor
                count: int = 0

                while low < high and count < spacing_allocation_details[4]:

                    spacing_points = low + (high - low) / 2
                    spacing_points = round(spacing_points, self.__limit)

                    if not direct_conversion_ctrl:
                        grid_points, hatch = self.grid_points_finder(num_workers, grid_bounds_full, info,
                                                                     spacing_points)
                    else:
                        grid_points = self.grid_points_finder(num_workers, grid_bounds_full, info, spacing_points,
                                                              direct_conversion_ctrl)

                    if not direct_conversion_ctrl:
                        num_points: int = len(grid_points[grid_points[:, :, -1] == self.__identity])
                    else:
                        num_points: int = len(grid_points)

                    percent = num_points / self.n * 100

                    print(f"\nSpacing - {spacing_points} and Percentage - {percent}%")

                    if spacing_allocation_details[2] <= percent < spacing_allocation_details[3]:
                        hatch_final = hatch.copy()
                        grid_points_final = grid_points.copy()
                        break

                    elif percent < spacing_allocation_details[2]:
                        high = spacing_points

                    else:
                        low = spacing_points

                    count += 1

                if count == spacing_allocation_details[4]:
                    hatch_final = hatch.copy()
                    grid_points_final = grid_points.copy()

                break

        if not direct_conversion_ctrl:
            return hatch_final, grid_points_final, spacing_points, percent
        else:
            return grid_points_final, spacing_points, percent

    def direct_gridder(self, data_set, grid_bounds_full, info, spacing):
        """
        Convert raw points to grid-aligned points.

        Parameters
        ----------
        data_set : iterable of numpy.ndarray
            Raw point data.
        grid_bounds_full : numpy.ndarray
            Axis bounds for grid.
        info : numpy.ndarray
            Grid configuration parameters.
        spacing : float
            Grid spacing distance.

        Returns
        -------
        set of tuple
            Unique grid-aligned points with identity label.
        """
        precision: int = ForceGridObject.count_decimal_places(spacing) - 1
        dim: int = int(info[3])
        grid_set: set = set()

        for point in data_set:

            point_copy = np.zeros(dim + 1)
            point_copy[-1] = self.__identity

            for coord in range(dim):

                point_copy[coord] = grid_bounds_full[coord, 0] + ForceGridObject.process(point[coord] - grid_bounds_full[coord, 0], spacing, precision)

            grid_set.add(tuple(point_copy))

        return grid_set
