# import statements
from Connectivity import Connectivity
from tqdm import tqdm
import time
import numpy as np


class ConnectivityDeterioration:
    """
    Simulate gradual deterioration of grid-based point clouds.

    ConnectivityDeterioration removes points from an object's grid
    representation over multiple iterations, tracking connectivity,
    point counts, and elapsed time.

    Attributes
    ----------
    __obj : list of numpy.ndarray
        Subsets of data points per mini-grid filtered by identity.
    __spacing : float
        Grid spacing loaded from 'Grid_Info.npy'.
    __num : int
        Number of deterioration iterations.
    __identity : int
        Identity label for selecting points.
    __probability_deter : float
        Probability of point removal per mini-grid (0 to 1).
    __delete_points_max : int
        Maximum number of points deletable per mini-grid.

    Methods
    -------
    connectivity_deterioration_analysis() -> None
        Execute deterioration simulation and save metrics.
    """

    def __init__(self, identity: int, obj: np.ndarray, x_points_num: int, delete_points_max: int, probability_deter: float = 0.05):
        """
        Initialize ConnectivityDeterioration.

        Parameters
        ----------
        identity : int
            Label of the object to deteriorate.
        obj : numpy.ndarray
            Grid-formatted data points.
        x_points_num : int
            Number of deterioration iterations.
        delete_points_max : int
            Max number of points deletable per mini-grid.
        probability_deter : float, optional
            Probability of deterioration per mini-grid (0 to 1), default=0.05.

        Raises
        ------
        FileNotFoundError
            If 'Grid_Info.npy' cannot be loaded.
        ValueError
            If `probability_deter` is not between 0 and 1.
        """

        self.__obj: list = list(obj.copy())

        for i, array in enumerate(self.__obj):

            self.__obj[i] = array[array[:, -1] == identity]

        info = np.load("Grid_Info.npy")
        self.__spacing = info[0]

        self.__num = x_points_num

        self.__identity = identity

        self.__probability_deter = probability_deter

        self.__delete_points_max = delete_points_max

    def connectivity_deterioration_analysis(self) -> None:
        """
        Execute deterioration simulation and save metrics.

        At each iteration, randomly remove points per mini-grid, then
        compute and record connectivity, object length, and elapsed time.

        Outputs
        -------
        Timings_Object_{identity}.npy : numpy.ndarray
            Elapsed time per iteration.
        Lengths_Object_{identity}.npy : numpy.ndarray
            Total point counts per iteration.
        CFS_Object_{identity}.npy : numpy.ndarray
            Connectivity factors per iteration.

        Returns
        -------
        None
        """

        start_time = time.time()

        timings = []
        obj_lengths = []
        cfs = []

        for i in tqdm(range(self.__num), desc="Calculating"):

            cfs.append(Connectivity.calc_connectivity_general(self.__obj, self.__spacing)[0])

            intermediate_length = 0

            for k in range(len(self.__obj)):
                intermediate_length += len(self.__obj[k])

            obj_lengths.append(intermediate_length)

            timings.append(time.time() - start_time)

            grid_selector = np.random.choice([0, 1], size=len(self.__obj), p=[1 - self.__probability_deter, self.__probability_deter])
            lengths = np.random.randint(0, self.__delete_points_max, len(self.__obj))

            for j, truth in enumerate(grid_selector):

                if truth == 1:

                    if lengths[j] < len(self.__obj[j]):

                        index_delete = np.random.randint(0, len(self.__obj[j]), lengths[j])
                        self.__obj[j] = np.delete(self.__obj[j], index_delete, axis=0)

                    else:

                        self.__obj[j] = np.asarray([])


        np.save(f"Timings_Object_{self.__identity}", np.asarray(timings))
        np.save(f"Lengths_Object_{self.__identity}", np.asarray(obj_lengths))
        np.save(f"CFS_Object_{self.__identity}", np.asarray(cfs))

        return None
