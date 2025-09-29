import numpy as np
from NCubeNeighbours import NCubeNeighbours
from Connectivity import Connectivity

class TopologicalDimension:
    """
    Compute topological dimension for fractals based on connectivity thresholds.

    This class loads grid settings from ``Grid_Info.npy`` and computes lower bound
    thresholds for connectivity factors to determine fractal dimensions.

    Attributes
    ----------
    __grid_spacing : float
        Grid spacing distance.
    __boundary_buffer : float
        Buffer distance for boundary extraction.
    __grid_divide : int
        Grid division count.
    __grid_dimension : int
        Number of grid dimensions.
    __cf_lower_bounds : list of float
        Connectivity factor lower bounds per dimension.

    Methods
    -------
    get_lower_bounds(dimension) -> list of float
        Compute empirical lower bounds for connectivity factors.
    get_lower_bounds_form(dimension) -> list of float
        Compute analytical lower bounds for connectivity factors.
    compute_topological_dimension(connectivity_factor) -> int
        Determine topological dimension given a connectivity factor.
    execute(basis: float = 50.0, degree: int = 2) -> None
        Run computation pipeline and save output results.
    """

    def __init__(self):
        """
        Initialize TopologicalDimension with grid parameters.

        Loads grid settings from "Grid_Info.npy" and computes connectivity factor
        lower bounds.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Instance attributes are initialized.
        """

        # Unpacking "Grid_Info.npy"
        info = np.load("Grid_Info.npy")  # loading grid info
        self.__grid_spacing = info[0]
        self.__boundary_buffer = info[1]
        self.__grid_divide = int(info[2])
        self.__grid_dimension = int(info[3])
        self.__cf_lower_bounds = TopologicalDimension.get_lower_bounds_form(self.__grid_dimension)

    @staticmethod
    def get_lower_bounds(dimension: int) -> list[float]:
        """
        Compute empirical lower bounds for connectivity factors.

        Parameters
        ----------
        dimension : int
            Number of grid dimensions to consider.

        Returns
        -------
        list of float
            Empirical lower bounds per dimension.
        """

        cf_lower_bounds = [0.0]

        for i in range(dimension):
            center = np.zeros(i + 2)
            center[-1] = -1

            structure_m_m = NCubeNeighbours(dimension=i + 1, center=center).get_ndcube_neighbours(1.0)

            structure_m_m_list = list(structure_m_m)
            structure_m_m_list.append(center)
            structure_m_m = np.array(structure_m_m_list)

            cfm_m = Connectivity.calc_connectivity_general([structure_m_m], 1.0, neighbour_set_method=False)[0]

            cfn_m = (pow(3, i + 1) - 1) / (pow(3, dimension) - 1) * cfm_m

            cf_lower_bounds.append(cfn_m)

        return cf_lower_bounds

    @staticmethod
    def get_lower_bounds_form(dimension: int) -> list[float]:
        """
        Compute analytical lower bounds for connectivity factors.

        Parameters
        ----------
        dimension : int
            Number of grid dimensions for analytical form.

        Returns
        -------
        list of float
            Analytical lower bounds per dimension.
        """

        cfn_m_list_form = [0.0]

        for i in range(1, dimension + 1):
            cfm_m = ((7 ** i) - (3 ** i)) / ((3 ** i - 1) * (3 ** i))
            cfn_m = (pow(3, i) - 1) / (pow(3, dimension) - 1) * cfm_m
            cfn_m_list_form.append(cfn_m)

        return cfn_m_list_form

    def compute_topological_dimension(self, connectivity_factor: float) -> int:
        """
        Determine topological dimension given a connectivity factor.

        Parameters
        ----------
        connectivity_factor : float
            Connectivity factor in [0.0, 1.0].

        Returns
        -------
        int
            Topological dimension index corresponding to the connectivity factor.

        Raises
        ------
        ValueError
            If connectivity_factor is not between 0 and 1.
        """

        if not 0 <= connectivity_factor <= 1:
            raise ValueError("Invalid Connectivity Factor (must be between 0 and 1)")

        for i in range(self.__grid_dimension):

            if self.__cf_lower_bounds[i] <= connectivity_factor < self.__cf_lower_bounds[i + 1]:

                return i

        return self.__grid_dimension

    def execute(self, basis: float = 50.0, degree: int = 2) -> None:
        """
        Run dimension computation pipeline and save results.

        Parameters
        ----------
        basis : float, optional
            Basis value for dimension computation (default: 50.0).
        degree : int, optional
            Degree parameter for fitting (default: 2).

        Returns
        -------
        None

        Notes
        -----
        Loads connectivity and weight files, computes topological dimensions,
        and saves results. Missing files are skipped gracefully.
        """

        try:
            cf_boundaries = np.load("Connectivity_Factors.npy")
            td_boundaries = []
            for cf in cf_boundaries:
                if cf >= 0:
                    td_boundaries.append(self.compute_topological_dimension(cf))
                else:
                    td_boundaries.append(-1)
            np.save("Topological_Dimensions_Boundaries.npy", td_boundaries)
        except FileNotFoundError:
            pass

        try:
            weights_boundaries = np.load("Weights_Boundaries.npy", allow_pickle=True)
            weighted_td_boundaries = []
            weights_boundaries_norm = []

            for weights in weights_boundaries:

                weights_iter = iter(weights.items())

                max_key = None
                max_value = -1.0

                sum_weights = 0.0

                weights_arr = []

                for key, value in weights_iter:

                    sum_weights += value

                    if value > max_value:
                        max_key = key
                        max_value = value

                    weights_arr.append(value)

                weights_arr = np.array(weights_arr)

                if sum_weights != 0:
                    weights_arr = weights_arr / sum_weights

                weights_boundaries_norm.append(weights_arr)

                weighted_td_boundaries.append(max_key)

            weights_boundaries_norm = np.array(weights_boundaries_norm)

            np.save("Weighted_Topological_Dimensions_Boundaries.npy", weighted_td_boundaries)
            np.save("Weights_Boundaries_Normalized.npy", weights_boundaries_norm)

        except FileNotFoundError:
            pass

        try:
            cf_hatch = np.load("Hatch_Connectivity_Factors.npy")
            td_hatch = []
            for cf in cf_hatch:
                if cf >= 0:
                    td_hatch.append(self.compute_topological_dimension(cf))
                else:
                    td_hatch.append(-1)
            np.save("Topological_Dimensions_Hatch.npy", td_hatch)
        except FileNotFoundError:
            pass

        try:
            weights_hatch = np.load("Weights_Hatch.npy", allow_pickle=True)
            weighted_td_hatch = []
            weights_hatch_norm = []

            for weights in weights_hatch:

                weights_iter = iter(weights.items())

                max_key = None
                max_value = -1.0

                sum_weights = 0.0

                weights_arr = []

                for key, value in weights_iter:

                    sum_weights += value

                    if value > max_value:
                        max_key = key
                        max_value = value

                    weights_arr.append(value)

                weights_arr = np.array(weights_arr)

                if sum_weights != 0:
                    weights_arr = weights_arr / sum_weights

                weights_hatch_norm.append(weights_arr)

                weighted_td_hatch.append(max_key)

            weights_hatch_norm = np.array(weights_hatch_norm)

            np.save("Weighted_Topological_Dimensions_Hatch.npy", weighted_td_hatch)
            np.save("Weights_Hatch_Normalized.npy", weights_hatch_norm)

        except FileNotFoundError:
            pass


        try:
            td_force = []
            cfs_o = []
            spacings = []
            analysis = np.load("Analysis.npy", allow_pickle=True)

            for i in range(len(analysis)):

                temp = np.array(analysis[i])
                temp = temp.astype(np.float64)  # Convert to proper numeric type
                x = temp[:, 4]
                y = temp[:, 3]
                z = np.log(temp[:, 2])

                # Split data at the turning point
                turning_index = np.argmax(y)  # Find the peak
                z_left, y_left = z[:turning_index + 1], y[:turning_index + 1]
                z_right, y_right = z[turning_index:], y[turning_index:]

                if basis >= x[turning_index]:
                    preference = 1
                else:
                    preference = 0

                coeffs = np.polyfit(x, y, degree)

                if len(y_left) != 1:
                    coeffs_spacing_left = np.polyfit(y_left, z_left, degree)
                else:
                    coeffs_spacing_left = -1.0

                if len(y_right) != 1:
                    coeffs_spacing_right = np.polyfit(y_right, z_right, degree)
                else:
                    coeffs_spacing_right = -1.0

                cf = 0.0

                for j, coeff in enumerate(coeffs):

                    cf += coeff * (basis ** (len(coeffs) - j - 1))

                spacing_left = 0.0
                spacing_right = 0.0

                if isinstance(coeffs_spacing_left, np.ndarray):

                    for j, coeff in enumerate(coeffs_spacing_left):
                        spacing_left += coeff * (cf ** (len(coeffs) - j - 1))

                else:

                    spacing_left = coeffs_spacing_left

                if isinstance(coeffs_spacing_right, np.ndarray):

                    for j, coeff in enumerate(coeffs_spacing_right):
                        spacing_right += coeff * (cf ** (len(coeffs) - j - 1))

                else:

                    spacing_right = coeffs_spacing_right

                spacing_left = np.e ** spacing_left
                spacing_right = np.e ** spacing_right

                cfs_o.append(cf)
                spacings.append(np.array([spacing_left, spacing_right, preference]))

                if cf >= 0:
                    td_force.append(self.compute_topological_dimension(cf))
                else:
                    td_force.append(-1)

            np.save("Topological_Dimensions_Object.npy", td_force)
            np.save("Object_Connectivity_Factors.npy", cfs_o)
            np.save("Force_Grid_Spacings.npy", spacings)

            try:

                weights_analysis = np.load("Weights_Analysis.npy", allow_pickle=True)
                weights_force_norm = []

                for obj in weights_analysis:

                    weights_perc_norm = []

                    for weights in obj:

                        weights_perc_arr = []

                        weights_sum = 0.0

                        weights_iter = iter(weights.items())

                        for key, value in weights_iter:

                            weights_sum += value

                            weights_perc_arr.append(value)

                        weights_perc_arr = np.array(weights_perc_arr)

                        if weights_sum != 0:
                            weights_perc_arr = weights_perc_arr / weights_sum

                        weights_perc_norm.append(weights_perc_arr)

                    weights_perc_norm = np.array(weights_perc_norm)

                    weights_force_norm.append(weights_perc_norm)

                vals = []

                for i, weights_perc_obj in enumerate(weights_force_norm):

                    coeffs_obj_weights = []

                    temp = np.array(analysis[i])
                    temp = temp.astype(np.float64)  # Convert to proper numeric type
                    x = temp[:, 4]

                    for j in range(len(weights_perc_obj[0])):

                        coeffs_obj_weights.append(np.polyfit(x, weights_perc_obj[:, j], degree))

                    vals_obj = []

                    coeffs_obj_weights = np.array(coeffs_obj_weights)

                    for coeffs in coeffs_obj_weights:

                        val = 0.0

                        for j, coeff in enumerate(coeffs):

                            val += coeff * (basis ** (len(coeffs) - j - 1))

                        vals_obj.append(val)

                    vals.append(vals_obj)

                vals = np.array(vals)

                weighted_td_force = np.argmax(vals, axis=1)

                np.save("Weighted_Topological_Dimensions_Force.npy", weighted_td_force)
                np.save("Weights_Force_Normalized.npy", vals)

            except FileNotFoundError:
                pass


        except FileNotFoundError:

            try:
                cf_force = np.load("Object_Connectivity_Factors.npy")
                td_force = []
                for cf in cf_force:
                    td_force.append(self.compute_topological_dimension(cf))
                np.save("Topological_Dimensions_Object.npy", td_force)
            except FileNotFoundError:
                pass

            try:
                weights_force = np.load("Weights_Force.npy", allow_pickle=True)
                weighted_td_force = []
                weights_force_norm = []

                for weights in weights_force:

                    weights_iter = iter(weights.items())

                    max_key = None
                    max_value = -1.0

                    sum_weights = 0.0

                    weights_arr = []

                    for key, value in weights_iter:

                        sum_weights += value

                        if value > max_value:
                            max_key = key
                            max_value = value

                        weights_arr.append(value)

                    weights_arr = np.array(weights_arr)

                    weights_arr = weights_arr / sum_weights

                    weights_force_norm.append(weights_arr)

                    weighted_td_force.append(max_key)

                weights_force_norm = np.array(weights_force_norm)

                np.save("Weighted_Topological_Dimensions_Force.npy", weighted_td_force)
                np.save("Weights_Force_Normalized.npy", weights_force_norm)

            except FileNotFoundError:
                pass
    
        return None
