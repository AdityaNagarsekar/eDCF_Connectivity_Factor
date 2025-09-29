# import statements
import numpy as np


class NCubeNeighbours:
    """
    Provide utilities to generate and manipulate n-dimensional cube neighbours around a point.

    Parameters
    ----------
    dimension : int
        Number of dimensions of the point.
    center : np.ndarray, shape (dimension,)
        Central point coordinates.

    Attributes
    ----------
    __dimension : int
        Dimension of the point space.
    __center : np.ndarray, shape (dimension,)
        Current center position.

    Methods
    -------
    get_neighbours(n, center, spacing, number) -> list
        Recursively generate neighbours for each intermediate dimension.
    fetch(neighbours, neighbours_array, number) -> None
        Flatten nested neighbour structures into a list.
    move(neighbours, move_to) -> None
        Translate existing neighbours relative to a new center.
    get_ndcube_neighbours(spacing, number=1) -> np.ndarray
        Return neighbours in an n-dimensional cube around the center.
    neighbours_in_set(points, spacing) -> np.ndarray
        Identify neighbours of the center within a given point set.

    Examples
    --------
    >>> gen = NCubeNeighbours(2, np.array([0.0, 0.0]))
    >>> neigh = gen.get_ndcube_neighbours(1.0)
    >>> neigh.shape
    (8, 2)
    """

    def __init__(self, dimension: int, center: np.ndarray):
        """
        Initialize NCubeNeighbours with target dimensionality and center point.

        Parameters
        ----------
        dimension : int
            Number of dimensions of the point space.
        center : np.ndarray, shape (dimension,)
            Coordinates of the central point.
        """

        self.__dimension = dimension
        self.__center = center.copy()
        self.__center_list = list(self.__center)

    def get_neighbours(self, n: int, center: list | np.ndarray, spacing: float, number: int) -> list:
        """
        Recursively generate neighbours by offsetting points along a specific dimension.

        Parameters
        ----------
        n : int
            Dimension index (1-indexed) to expand (n > 0).
        center : list or numpy.ndarray
            Nested list or array of point coordinates at previous dimension.
        spacing : float
            Distance step along the nth dimension.
        number : int
            Half-range of offsets to generate per dimension (e.g., 1 for immediate neighbours).

        Returns
        -------
        list of numpy.ndarray
            Updated point arrays with offsets applied in the nth dimension.
        """

        updated_center = []  # updated_center initialization to an empty list

        if not isinstance(center, np.ndarray):  # stopping condition is that we bump into a numpy array

            for j in range(-number, number + 1):  # parsing through the divisions in the current center
                updated_center.append(self.get_neighbours(n, center[j + number].copy(), spacing,
                                                          number))  # reduce the dimension by taking a division of center and pass it recursively.

        else:

            for i in range(-number,
                           number + 1):  # parsing through divisions of array with numpy array as the current center so the dimensional change is technically 0 to 1
                temp = center.copy()  # making a copy of the array.
                temp[
                    n - 1] += i * spacing  # adding a line across the nth dimension in each point with the center as a point.
                updated_center.append(temp)  # updating the center.

        return updated_center

    def fetch(self, neighbours: list | np.ndarray, neighbours_array: list, number: int) -> None:
        """
        Flatten nested neighbour structure into a flat list.

        Parameters
        ----------
        neighbours : list or numpy.ndarray
            Nested list/array of neighbour points to flatten.
        neighbours_array : list
            Collector list to append flattened neighbour arrays.
        number : int
            Half-range of offsets (e.g., 1 yields immediate neighbours).

        Returns
        -------
        None

        Notes
        -----
        After flattening, `neighbours_array` will contain numpy arrays which may include the center
        and should be filtered out externally if needed.
        """

        for i in range(
                2 * number + 1):  # we are parsing through 2n + 1 structures which is dependent on number. This is since we are taking this number of divisions on -ve and +ve side and including 0 as well.

            if not isinstance(neighbours[i],
                              np.ndarray):  # if we do not have a numpy array then continue going deeper in the recursive structure using the same dimensionality reduction as get_neighbours

                self.fetch(neighbours[i], neighbours_array, number)

            else:

                neighbours_array.append(
                    neighbours[i])  # when we hit a numpy array leaf then we append that to the array.

        return None

    def move(self, neighbours: np.ndarray, move_to: np.ndarray) -> None:
        """
        Translate neighbour points relative to a new central point.

        Parameters
        ----------
        neighbours : numpy.ndarray, shape (n_neighbours, dimension)
            Array of neighbour point coordinates to translate.
        move_to : numpy.ndarray, shape (dimension,)
            New central point coordinates.

        Returns
        -------
        None

        Notes
        -----
        Computes translation vector as `move_to - old_center` and applies it to all neighbours.
        Updates internal center to `move_to`.
        """

        moving_by = np.concatenate((move_to[:-1] - self.__center[:-1], np.asarray([0.0])))

        for i in range(len(neighbours)):
            neighbours[i] = neighbours[i] + moving_by

        self.__center = move_to

        return None

    def get_ndcube_neighbours_deprecated(self, spacing: float, number: int = 1) -> np.ndarray:
        """
        Generate all n-dimensional cubic neighbours around the center point.

        Parameters
        ----------
        spacing : float
            Distance between adjacent neighbour points.
        number : int, optional
            Half-range of neighbour offsets per axis (default: 1 for immediate neighbours).

        Returns
        -------
        numpy.ndarray, shape (n_neighbours, dimension)
            Coordinates of neighbour points excluding the center point.
        """

        neigh: np.ndarray | list = self.__center.copy()  # passing a copy of the center here for generating the neighbours.

        for i in range(1, self.__dimension + 1):  # parsing through dimensions increasing with each iteration
            neigh = self.get_neighbours(i, neigh.copy(), spacing, number)

        neighbours: list = []  # crucial assignment of empty list for readable array format conversion

        self.fetch(neigh, neighbours, number)

        neighbours = [point for point in neighbours if
                      not np.array_equal(point, self.__center)]  # removing the center to get true neighbours

        return np.asarray(neighbours)

    def neighbours_in_set(self, points: np.ndarray, spacing: int) -> np.ndarray:
        """
        Filter a point set to identify direct neighbours around the center.

        Parameters
        ----------
        points : numpy.ndarray, shape (n_points, dimension)
            Array of candidate points to filter.
        spacing : int
            Grid spacing threshold; float spacing may introduce precision issues.

        Returns
        -------
        numpy.ndarray, shape (n_neighbours, dimension)
            Points adjacent to the internal center within ±spacing on each axis.
        """

        # Initially everything is true based on the fact that all need to be evaluated.
        mask_left = np.full(len(points), True)  # for neighbours to the left of axis(negative).
        mask_center = np.full(len(points), True)  # for neighbours on the axis
        mask_right = np.full(len(points), True)  # for neighbours to the right of the axis(positive)
        mask_for_center = np.full(len(points), True)  # to track down the center simultaneously and specifically

        mask = np.logical_or.reduce([mask_left, mask_right, mask_center])  # combined mask of neighbours along the axis

        center_index_mask = np.where(mask)[0]  # center_index tracker for mask

        center_index = center_index_mask  # center_index tracker

        indices_mask = np.where(mask)[0]  # indices tracker for mask
        indices = indices_mask  # indices tracker

        for i in range(self.__dimension):  # parsing through dimensions

            mask_left = (points[indices_mask, i] == self.__center[i] + (-1 * spacing))  # tracking negative neighbours
            mask_center = (points[indices_mask, i] == self.__center[i] + (0 * spacing))  # tracking on axis neighbours
            mask_right = (points[indices_mask, i] == self.__center[i] + (1 * spacing))  # tracking positive neighbours

            if len(center_index_mask) != 0:  # checking if center can still exist
                mask_for_center = (points[center_index_mask, i] == self.__center[i] + (
                            0 * spacing))  # if it can then continue tracking

            mask = np.logical_or.reduce([mask_left, mask_right, mask_center])  # combining tracked neighbours of an axis

            if not np.any(mask):  # if we don't have any neighbours possible then return empty
                return np.array([], dtype=int)

            indices_mask = np.where(mask)[0]  # updating indices tracker for mask

            if np.any(
                    mask_for_center):  # if there is still a possibility for center to exist based on tracked values then continue traversing for center
                center_index_mask = np.where(mask_for_center)[0]
                center_index = center_index[center_index_mask]
                center_index_mask = center_index
            else:
                center_index = np.array([], dtype=int)
                center_index_mask = center_index

            indices = indices[indices_mask]  # indices tracker updated based on indices tracker for mask.

            indices_mask = indices  # updating the indices tracker for mask for next dimension

        if len(center_index) != 0:  # if we have a center index then
            indices = np.setdiff1d(indices, center_index)
            return points[indices]
        else:
            return points[indices]  # in case center is not present in set from the start

    def get_ndcube_neighbours(self, spacing: float, number: int = 1):

        accumulator = []
        res = []

        def helper(index: int):

            if index == self.__dimension:
                if accumulator != self.__center_list:
                    final = accumulator + self.__center_list[self.__dimension:]
                    res.append(final)
                return

            for i in range(-number, number + 1):
                accumulator.append(self.__center[index] + i * spacing)
                helper(index + 1)
                accumulator.pop()

        helper(0)

        return np.array(res)