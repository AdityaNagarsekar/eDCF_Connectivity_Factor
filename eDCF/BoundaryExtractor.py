import numpy as np
import multiprocessing

import psutil

from NCubeNeighbours import NCubeNeighbours

class BoundaryExtractor:
    """
    Extract boundary points between labeled grid structures using NCubeNeighbours.

    Processes labeled grid data from "Grid_Points.npy" to identify inter-structure boundaries and saves results to "Neighbour_Boundary.npy".

    Methods
    -------
    mini_extract_neighbour(points, weight=0.5, neighbours_set_method=False) -> list
        Compute boundary points within a mini-grid segment.
    get_available_cores(threshold=50) -> int
        Determine CPU cores available for computation.
    extract_neighbour(weight=0.5, neighbours_set_method=False) -> None
        Extract boundaries for the entire grid and save to file.
    fetch(mini_grid_num=-1, lower_struct=1, higher_struct=2, filter_boundary=False) -> list
        Retrieve computed boundaries for specified block and structure indices.

    Notes
    -----
    Input file: "Grid_Points.npy"
    Output file: "Neighbour_Boundary.npy"

    Examples
    --------
    >>> BoundaryExtractor.extract_neighbour()
    >>> b = BoundaryExtractor.fetch(mini_grid_num=0, lower_struct=1, higher_struct=2)
    """

    @staticmethod
    def mini_extract_neighbour(points: np.ndarray, weight: float = 0.5, neighbours_set_method: bool = False) -> list:
        """
        Compute boundary points for a mini-grid segment between labelled structures.

        Parameters
        ----------
        points : numpy.ndarray, shape (n_points, n_dims+1)
            Labelled grid points for the mini-grid block; last column contains structure IDs.
        weight : float, optional
            Interpolation weight for boundary point calculation (default: 0.5).
        neighbours_set_method : bool, optional
            Use `neighbours_in_set` when True; otherwise use `get_ndcube_neighbours` (default: False).

        Returns
        -------
        list of list of numpy.ndarray
            Nested list [low][high] of boundary point arrays between each pair of structures.

        Notes
        -----
        - Labels in `points[:, -1]` are expected to be integer IDs starting at 1.
        - Boundary points have label -1 in the last column.

        Examples
        --------
        >>> mini_block = np.load("Grid_Points.npy")[0]
        >>> boundaries = BoundaryExtractor.mini_extract_neighbour(mini_block, weight=0.3)
        """

        grid_data_l = []  # to separate the grid_data into list based on labels
        # NOTE: DO NOT TRY TO CONVERT THIS TO NUMPY ARRAY SINCE THE LIST IS INHOMOGENEOUS

        i: int = 0  # loop control variable
        num_struct = len(np.load("Datapoints.npy", allow_pickle=True))  # we are getting the number of structures present in data_points

        while i < num_struct:  # parsing all the structures

            grid_data_i = points[
                points[:, -1] == i + 1]  # extracting the grid data  with label of (i + 1) th structure
            grid_data_l.append(grid_data_i)  # appending the data to list of grid data
            i += 1  # update statement

        boundary_mini_grid = []  # will contain list of boundary points

        low: int = 0  # lower structure index
        high: int = 1  # higher structure index

        info = np.load("Grid_Info.npy")  # taking out grid info

        spacing = info[0]  # fetching spacing between each point in grid

        # precision = int(math.ceil(abs(math.log10(spacing))))  # precision rounding is required as the numbers show variation at 15 decimal place due to how floating point numbers are stored in a computer. (hardware limitations)
        precision = BoundaryExtractor.count_decimal_places(spacing)

        if not neighbours_set_method:

            points = {tuple(round(coord, precision) for coord in point) for point in points}  # precision rounding of all the points is required.

        else:

            for i, grid_data in enumerate(grid_data_l):
                grid_data_l[i] = np.asarray([[int(round((pow(10, precision) * coord))) for coord in point] for point in grid_data])

        while low < num_struct - 1:  # parsing through the structures

            boundary_low = []  # holding the boundary points for a lower structure with several higher structures (e.g. if we have identity labels 1, 2, 3 then 1 will contain the boundary between 1-2 and 1-3)

            while high < num_struct:  # parsing through structures above lower structures

                boundary_high = []  # contains boundary between current low and current high structs

                if len(grid_data_l[low]) != 0 and len(grid_data_l[high]) != 0:  # only do this if both lower structure and higher structure points are present in the mini grid.

                    if len(grid_data_l[low]) <= len(grid_data_l[high]):  # taking the structure with lower number of points.

                        n_cube = NCubeNeighbours(len(grid_data_l[low][0]) - 1, grid_data_l[low][0])

                        neighbours = []

                        for point1 in grid_data_l[low]:  # iterating through points which belong to lower circle when lower structure points are lesser.

                            if not neighbours_set_method:

                                if len(neighbours) == 0:

                                    neighbours = n_cube.get_ndcube_neighbours(spacing)  # getting neighbours of this specific point.

                                else:

                                    neighbours = np.array(list(neighbours))
                                    n_cube.move(neighbours, move_to=point1)

                                neighbours[:, -1] = high + 1  # setting the label of the higher structure to neighbours

                                neighbours = {tuple(round(coord, precision) for coord in point) for point in neighbours}  # getting precision values of neighbours.
                                common_points = neighbours.intersection(points)  # taking intersection between neighbours and grid points.

                                common_points = np.array(list(common_points))  # getting to numpy array format

                                for point2 in common_points:  # iterating through common points to compute boundary points.

                                    b_point = ((1 - weight) * point1) + (weight * point2)  # weighted boundary point calculation
                                    b_point[-1] = -1  # setting label of boundary points to -1.0
                                    boundary_high.append(b_point)  # found a boundary point

                            else:

                                neighbours = NCubeNeighbours(len(point1) - 1, point1).neighbours_in_set(grid_data_l[high], int((pow(10, precision) * spacing)))

                                for point2 in neighbours:  # iterating through neighbours to compute boundary points.

                                    b_point = (((1 - weight) * point1) + (
                                                weight * point2)) * pow(10, -precision) # weighted boundary point calculation
                                    b_point[-1] = -1  # setting label of boundary points to -1.0
                                    boundary_high.append(b_point)  # found a boundary point

                    else:  # same as above but for higher structure instead of lower structure (can put this code block in a function to avoid repetition of code)

                        # ----------

                        for point1 in grid_data_l[high]:

                            if not neighbours_set_method:
                                neighbours = NCubeNeighbours(len(point1) - 1, point1).get_ndcube_neighbours(spacing)
                                neighbours[:, -1] = low + 1

                                neighbours = {tuple(round(coord, precision) for coord in point) for point in neighbours}

                                common_points = neighbours.intersection(points)

                                common_points = np.array(list(common_points))

                                for point2 in common_points:
                                    b_point = (weight * point1) + ((1 - weight) * point2)
                                    b_point[-1] = -1
                                    boundary_high.append(b_point)
                            else:

                                neighbours = NCubeNeighbours(len(point1) - 1, point1).neighbours_in_set(
                                    grid_data_l[low], int((pow(10, precision) * spacing)))

                                for point2 in neighbours:
                                    b_point = ((weight * point1) + ((1 - weight) * point2)) * pow(10, -precision)
                                    b_point[-1] = -1
                                    boundary_high.append(b_point)

                        # ----------

                if len(boundary_high) == 0:  # if we do not find a boundary point at all in this grid between the two structures from the if condition before
                    boundary_high = [np.array([])]  # an empty numpy array assignment but in a list to maintain structure of 2D numpy array for points
                else:
                    boundary_high = np.asarray(boundary_high)  # normal conversion to numpy array

                boundary_low.append(boundary_high)  # appending all the boundary high to this to store for structures

                high += 1  # updating high till number of structures

            boundary_mini_grid.append(boundary_low)  # appending a lower-higher system of boundary points to the grid boundaries

            low += 1  # low is updated by 1
            high = low + 1  # low is updated to the next structure

        return boundary_mini_grid

    @staticmethod
    def get_available_cores(threshold: int = 50) -> int:
        """
        Determine CPU cores available for computation based on current usage.

        Parameters
        ----------
        threshold : int, optional
            CPU usage percentage threshold at which a core is considered busy (default: 50).

        Returns
        -------
        int
            Number of CPU cores available for use.
        """
        num_cores: int = multiprocessing.cpu_count()
        current_usage = psutil.cpu_percent(interval=1)

        # If current CPU usage is above the threshold, reduce the number of available cores
        available_cores: int = num_cores
        if current_usage > threshold:
            # Reduce the number of available cores based on the usage
            available_cores = max(1, num_cores - int(current_usage / 100 * num_cores))

        return available_cores

    @staticmethod
    def extract_neighbour(weight: float = 0.5, neighbours_set_method: bool = False) -> None:
        """
        Extract and save boundary points for the entire grid across all mini-grid segments.

        Parameters
        ----------
        weight : float, optional
            Interpolation weight for boundary calculation between structures (default: 0.5).
        neighbours_set_method : bool, optional
            Use `neighbours_in_set` when True; otherwise use `get_ndcube_neighbours` (default: False).

        Returns
        -------
        None

        Notes
        -----
        Input file: "Grid_Points.npy"
        Output file: "Neighbour_Boundary.npy"

        Examples
        --------
        >>> BoundaryExtractor.extract_neighbour(weight=0.3)
        """

        # Get the number of available CPU cores based on current usage
        num_workers = BoundaryExtractor.get_available_cores(threshold=50)

        grid_points = np.load("Grid_Points.npy")  # loading grid points from 'Grid_points.npy' file.

        grid_points_list = []  # initializing a list to convert 3D numpy array to a list with 2D numpy arrays to enable multiprocessing.
        weights = []  # initializing a list of weights which are the same here.
        neighbours_set_methods = []

        # preparation for multiprocessing

        # ----------

        for i in range(len(grid_points)):

            grid_points_list.append(grid_points[i])
            weights.append(weight)
            neighbours_set_methods.append(neighbours_set_method)

        params = list(zip(grid_points_list, weights, neighbours_set_methods))

        # Create a Pool with the calculated number of available CPU cores
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use pool.starmap to unpack the parameter tuples
            neighbour_boundary = pool.starmap(BoundaryExtractor.mini_extract_neighbour, params)

        # ----------

        object_array = np.array(neighbour_boundary, dtype=object)  # creation of object array from list for storage.
        np.save("Neighbour_Boundary", object_array)

        return None

    @staticmethod
    def fetch(mini_grid_num: int = -1, lower_struct: int = 1, higher_struct: int = 2, filter_boundary: bool = False) -> list:
        """
        Retrieve computed boundary points for specified grid block and structure pair.

        Parameters
        ----------
        mini_grid_num : int, optional
            Index of the mini-grid block to fetch (-1 for all blocks, default: -1).
        lower_struct : int, optional
            Lower structure ID (1-indexed).
        higher_struct : int, optional
            Higher structure ID (must be > lower_struct).
        filter_boundary : bool, optional
            If True, remove empty boundary arrays (default: False).

        Returns
        -------
        list of numpy.ndarray
            Boundary point arrays. If `mini_grid_num` is -1, returns list of arrays across all blocks; else a single array.

        Notes
        -----
        - Data loaded from 'Neighbour_Boundary.npy'.
        - Boundary arrays have label -1 in the last column.
        """

        neighbour_boundary = np.load("Neighbour_Boundary.npy", allow_pickle=True)  # loading the boundary system from the file 'Neighbour_Boundary.npy'

        boundary = []  # initializing an empty list to hold boundary points for returning a specific boundary.

        if mini_grid_num == -1:

            for i in range(len(neighbour_boundary)):  # iterating through mini grids

                boundary.append(neighbour_boundary[i][lower_struct - 1][higher_struct - lower_struct - 1])  # extracting of the specific boundary points and appending it to our list

        else:

            boundary = neighbour_boundary[mini_grid_num][lower_struct - 1][higher_struct - lower_struct - 1]  # getting boundary for only for 1 specific mini grid

        if filter_boundary:  # if we are filtering the boundary

            filtered = []  # to store the filtered boundary

            for bound in boundary:  # iterating through boundary

                if len(bound[0]) != 0:  # getting rid of empty numpy arrays.
                    filtered.append(bound)

            boundary = filtered  # filtered reference is now passed to boundary.

        return boundary

    @staticmethod
    def count_decimal_places(number: float, max_to_consider_trailing: int = 8) -> int:
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
