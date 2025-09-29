# import statements
import math
import multiprocessing

import psutil

from BoundaryExtractor import BoundaryExtractor
from FractalDetector import FractalDetector
from NCubeNeighbours import NCubeNeighbours
import numpy as np
from typing import Any
from Weight import Weight

class Connectivity:
    """
    Compute connectivity factor for spatial point sets on a grid.

    The connectivity factor quantifies neighbor overlap proportion, yielding values in [0.0, 1.0].

    Methods
    -------
    __init__(points, spacing, neighbour_set_method=False)
    calculate_connectivity_factor(points)
    get_available_cores(threshold=50)
    calc_connectivity(spacing, neighbour_set_method=False)
    calc_connectivity_boundary(boundary, spacing, neighbour_set_method=False)
    calc_connectivity_general(points, spacing, neighbour_set_method=False)

    Examples
    --------
    >>> conn = Connectivity(points, spacing=0.1)
    >>> factor, count, weight = conn.calculate_connectivity_factor(points)
    >>> Connectivity.calc_connectivity(spacing=0.1)
    """

    def __init__(self, points, spacing, neighbour_set_method: bool = False):
        """
        Initialize Connectivity with point set and grid spacing.

        Parameters
        ----------
        points : numpy.ndarray, shape (n_points, n_dims+1)
            Input points including labels in the last column.
        spacing : float
            Grid spacing distance for neighbor search.
        neighbour_set_method : bool, optional
            If True, use array-based neighbor algorithm; set-based otherwise (default).

        Attributes
        ----------
        spacing : float
        precision : int
        dimension : int
        neighbour_set_method : bool
        points : numpy.ndarray or None
        points_tup : set or None
        """

        self.spacing = spacing
        # self.precision = int(math.ceil(abs(math.log10(spacing))))
        self.precision = Connectivity.count_decimal_places(spacing)# calculation of precision for precision rounding

        self.dimension = len(points[0, :-1])  # getting the dimension of the points. Requires points to have a format of [x, y, z, ... n dimensions, label]
        self.neighbour_set_method: bool = neighbour_set_method

        if neighbour_set_method:
            self.points = (np.round(points[:, :-1], decimals=self.precision) * pow(10, self.precision)).astype(int)
        else:
            self.points_tup = {tuple(round(coord, self.precision) for coord in point) for point in points}  # precision rounding and conversion to tuples for set intersection use

    @staticmethod
    def count_decimal_places(number, max_to_consider_trailing: int = 8):
        """
        Count the number of decimal places in a floating-point number.

        Parameters
        ----------
        number : float
            The number to analyze.
        max_to_consider_trailing : int, optional
            Maximum number of trailing decimal places to consider (default: 8).

        Returns
        -------
        int
            Number of decimal places in the number.
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

    def calculate_connectivity_factor(self, points: np.ndarray):
        """
        Compute connectivity factor for a given point array.

        Parameters
        ----------
        points : numpy.ndarray, shape (n_points, n_dims)
            Coordinates of points to evaluate.

        Returns
        -------
        tuple of (float, int, Weight)
            connectivity_factor : float
            count : int
                Number of points evaluated.
            weight : Weight
                Weight object summarizing neighbor stats.
        """

        final_frac = 0.0  # final_frac starts off as 0.0

        wt = Weight(self.dimension)

        n_cube = NCubeNeighbours(self.dimension, points[0])
        neighbours = []

        for point in points:  # iterating through the points

            if not self.neighbour_set_method:

                if len(neighbours) == 0:
                    neighbours = n_cube.get_ndcube_neighbours(self.spacing)  # getting the neighbours of each point
                else:
                    neighbours = np.array(list(neighbours))
                    point = np.asarray(tuple(round(coord, self.precision) for coord in point))
                    n_cube.move(neighbours, move_to=point)

                n_neighbours = len(neighbours)  # getting the number of neighbours for each point (technically not required everytime and can be stored in shared space instead).

                neighbours = {tuple(round(coord, self.precision) for coord in point) for point in neighbours}  # getting the precision rounded neighbour set of tuples

                common_points = neighbours.intersection(self.points_tup)  # getting intersection between the global set and the neighbour set.

                common_points = np.array(list(common_points))  # the common points are converted to a numpy array for convenience (this step is avoidable)

                fraction = len(common_points) / n_neighbours  # we take the length of common points here seeing how many neighbours occur in global set essentially and divide by the cardinality of the neighbour set.

                wt.add_point_info(len(common_points))

            else:

                point = (np.round(point[:-1], decimals=self.precision) * pow(10, self.precision)).astype(int)

                neighbours = NCubeNeighbours(self.dimension, point).neighbours_in_set(self.points, int((pow(10, self.precision) * self.spacing)))  # getting neighbours of this specific point.

                fraction = len(neighbours) / (pow(3, self.dimension) - 1)

            final_frac += fraction  # here we update the final fraction continuously

        final_frac /= len(points)  # finally we divide by the number of points to get an average (this step is avoidable but for that we have to modify the calc_connectivity_boundary and calc_connectivity_general functions a bit)

        return final_frac, len(points), wt

    @staticmethod
    def get_available_cores(threshold: int = 50):
        """
        Determine available CPU cores based on system CPU usage.

        Parameters
        ----------
        threshold : int, optional
            CPU usage percent threshold (default: 50).

        Returns
        -------
        int
            Number of CPU cores considered available.
        """
        num_cores = multiprocessing.cpu_count()
        current_usage = psutil.cpu_percent(interval=1)

        # If current CPU usage is above the threshold, reduce the number of available cores
        available_cores = num_cores
        if current_usage > threshold:
            # Reduce the number of available cores based on the usage
            available_cores = max(1, num_cores - int(current_usage / 100 * num_cores))

        return available_cores

    @staticmethod
    def calc_connectivity(spacing: float, neighbour_set_method: bool = False) -> None:
        """
        Compute connectivity factors for all boundary pairs and save to files.

        Parameters
        ----------
        spacing : float
            Grid spacing for neighbor search.
        neighbour_set_method : bool, optional
            Use array-based neighbor algorithm if True (default: False).

        Notes
        -----
        Requires 'Boundary_Type.npy' and boundary data from BoundaryExtractor.

        Outputs
        -------
        Connectivity_Factors.npy : numpy.ndarray, shape (n_pairs,)
            Connectivity factors per boundary pair.
        Weights_Boundaries.npy : numpy.ndarray
            Corresponding weight summaries per boundary.
        """

        boundaries: list = []  # initializing an empty list to store boundaries
        try:
            boundary_type = np.load("Boundary_Type.npy")  # checking for existence
            info = np.load("Grid_Info.npy")
        except FileNotFoundError:
            FractalDetector().detect(create_boundary_type_only=True)  # creation of only the required file.
            boundary_type = np.load("Boundary_Type.npy")
            info = np.load("Grid_Info.npy")
        
        dim = info[3]

        for i in range(len(boundary_type)):  # parsing through structures.

            bound_t = boundary_type[i]  # getting the current boundary type

            boundaries.append(BoundaryExtractor.fetch(mini_grid_num=-1, lower_struct=bound_t[0], higher_struct=bound_t[1], filter_boundary=True))  # appending the boundary from the boundary type.

        connectivity_factors = []  # connectivity factor list is initialized as empty
        weights_bounds = []

        for boundary in boundaries:  # parsing through boundaries

            intermediate = Connectivity.calc_connectivity_boundary(boundary, spacing, neighbour_set_method)

            if intermediate == -1:
                intermediate = [-1.0, Weight(int(dim))]

            connectivity_factors.append(intermediate[0])  # appending connectivity factor for current boundary.
            weights_bounds.append(intermediate[1].weights)

        connectivity_factors = np.asarray(connectivity_factors)
        weights_bounds = np.array(weights_bounds)

        np.save("Connectivity_Factors.npy", connectivity_factors)
        np.save("Weights_Boundaries.npy", weights_bounds)

    @staticmethod
    def calc_connectivity_boundary(boundary: list, spacing: float, neighbour_set_method: bool = False):
        """
        Compute connectivity factor for a single boundary dataset in parallel.

        Parameters
        ----------
        boundary : list of numpy.ndarray
            List of point arrays forming the boundary.
        spacing : float
            Grid spacing for neighbor search.
        neighbour_set_method : bool, optional
            Select neighbor algorithm (default: False).

        Returns
        -------
        tuple of (float, Weight)
            connectivity_factor : float
            weight : Weight
        """

        # Get the number of available CPU cores based on current usage
        num_workers = Connectivity.get_available_cores(threshold=50)

        if len(boundary) == 0:
            return -1

        boundary_arr = np.vstack(boundary)

        boundary_arr = np.array(list({tuple(point) for point in boundary_arr}), dtype=float)
        if len(boundary_arr) > num_workers:
            dim = len(boundary_arr[0])
            third_axis = num_workers
            while third_axis < len(boundary_arr):
                if len(boundary_arr) % (dim * third_axis) == 0:
                    break
                else:
                    third_axis += 1

            boundary = list(boundary_arr.reshape(third_axis, -1, dim))

        c = Connectivity(boundary_arr, spacing, neighbour_set_method)

        params = list(zip(boundary))

        # Create a Pool with the calculated number of available CPU cores
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use pool.starmap to unpack the parameter tuples
            frac_weights = pool.starmap(c.calculate_connectivity_factor, params)

        final_frac = 0.0
        full_weight = 0

        wt = Weight(len(boundary_arr[0]) - 1)

        for i in range(len(frac_weights)):

            final_frac += frac_weights[i][0] * frac_weights[i][1]
            full_weight += frac_weights[i][1]
            wt + frac_weights[i][2]

        final_frac = final_frac / full_weight

        return final_frac, wt

    @staticmethod
    def calc_connectivity_general(points: list, spacing: float, neighbour_set_method: bool = False):
        """
        Compute connectivity factor for a general point list.

        Parameters
        ----------
        points : list of numpy.ndarray
            Grouped point arrays to evaluate connectivity.
        spacing : float
            Grid spacing for neighbor search.
        neighbour_set_method : bool, optional
            Select neighbor algorithm (default: False).

        Returns
        -------
        tuple of (float, Weight) or -1.0
            connectivity_factor and Weight, or -1.0 for invalid input.

        Notes
        -----
        Multiprocessing currently disabled; improvements pending.
        """

        # Get the number of available CPU cores based on current usage
        num_workers: int = Connectivity.get_available_cores(threshold=50)

        if len(points) == 0:  # if there are no points returning -1 to signal wrong input.
            return -1

        # Remove empty arrays
        points = [array for array in points if len(array) != 0]  # removing the empty arrays if any.

        points_arr = np.vstack(points)  # stacking the points to make a global set of structure points using the __init__ in the next step.

        points_arr = np.array(list({tuple(point) for point in points_arr}), dtype=float)
        if len(points_arr) > num_workers:
            dim = len(points_arr[0])
            third_axis = num_workers
            while third_axis < len(points_arr):
                if len(points_arr) % (dim * third_axis) == 0:
                    break
                else:
                    third_axis += 1

            points = list(points_arr.reshape(third_axis, -1, dim))

        c = Connectivity(points_arr, spacing, neighbour_set_method)  # creation of Connectivity object.

        # preparing for multiprocessing

        # ----------

        if neighbour_set_method:
            params = list(zip(points))  # NOTE: for some reason without the neighbour_set_method, we can't multiprocess as a result of which it is in fact slower but without multiprocessing and using the NCubeNeighbours original algorithm is way faster.
        else:
            params = list(zip([points_arr]))  # FIXME: for some reason multiprocessing is not working with the ncubeneighbours method

            # TODO: to enable multiprocessing again please use copy the statement from if neighbour_set_method block or remove the if part and keep the statement in if block only.

        # # Create a Pool with the calculated number of available CPU cores
        # with multiprocessing.Pool(processes=num_workers) as pool:
        #     # Use pool.starmap to unpack the parameter tuples
        #     frac_weights = pool.starmap(c.calculate_connectivity_factor, params)

        # # ----------

        frac_weights = [c.calculate_connectivity_factor(points_arr)]  # FIXME: multiprocessing is not working 

        final_frac: float = 0.0  # before the final_fac is 0
        full_weight: int = 0  # the weight is also assigned to 0

        wt = Weight(len(points_arr[0]) - 1)

        # adding the final_fracs and total weights for combined average.
        for i in range(len(frac_weights)):
            final_frac += frac_weights[i][0] * frac_weights[i][1]
            full_weight += frac_weights[i][1]
            wt + frac_weights[i][2]

        final_frac = final_frac / full_weight  # getting the connectivity factor of the structure.

        return final_frac, wt

    @staticmethod
    def add_potential(point: np.ndarray,  connectivity_set: set[tuple[Any, ...]] , connectivity_current: float, spacing: float, neighbour_set_method: bool = False) -> float:
        """
        Add connectivity potential to a point based on its neighbors.

        Parameters
        ----------
        point : numpy.ndarray
            Point coordinates.
        connectivity_set : set of tuple
            Set of connectivity tuples.
        connectivity_current : float
            Current connectivity value.
        spacing : float
            Grid spacing for neighbor search.
        neighbour_set_method : bool, optional
            Use array-based neighbor algorithm if True (default: False).

        Returns
        -------
        float
            Updated connectivity value.
        """

        final_frac = connectivity_current  # final_frac starts off as 0.0

        n_cube = NCubeNeighbours(len(point) - 1, point)

        preci = int(math.ceil(abs(math.log10(spacing))))

        if not neighbour_set_method:
            pass

        neighbours = n_cube.get_ndcube_neighbours(spacing)  # getting the neighbours of each point

        n_neighbours = len(neighbours)  # getting the number of neighbours for each point (technically not required everytime and can be stored in shared space instead).

        neighbours = {tuple(round(coord, preci) for coord in point) for point in neighbours}  # getting the precision rounded neighbour set of tuples

        common_points = neighbours.intersection(connectivity_set)  # getting intersection between the global set and the neighbour set.

        common_points = np.array(list(common_points))  # the common points are converted to a numpy array for convenience (this step is avoidable)

        fraction = len(common_points) / n_neighbours  # we take the length of common points here seeing how many neighbours occur in global set essentially and divide by the cardinality of the neighbour set.

        # else:
        #
        #     point = (np.round(point[:-1], decimals=preci) * pow(10, preci)).astype(int)
        #
        #     neighbours = NCubeNeighbours(len(point) - 1, point).neighbours_in_set(self.points, int((pow(10,
        #                                                                                                 self.precision) * self.spacing)))  # getting neighbours of this specific point.
        #
        #     fraction = len(neighbours) / (pow(3, self.dimension) - 1)
        #

        final_frac = (final_frac * len(connectivity_set) + 2 * fraction) /  (len(connectivity_set) + 1) # here we update the final fraction continuously

        return final_frac