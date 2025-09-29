import math
import os
import shutil
from time import time, sleep
from tqdm import tqdm
from typing import Tuple
from deprecated import deprecated

import numpy as np


@deprecated(version="2.4.2", reason="Scheduled for revision; may not function correctly in current releases")
class TimeAnalysis:
    """
    .. deprecated:: 2.4.2
        Outdated implementation; functionality may not work in current releases. Scheduled for revision in a future version.

    Profile execution time of driver function components.

    Executes the provided driver and logs durations for each control step across grid scaling variants.

    Methods
    -------
    analyse(driver, scale_grid_space='multiply', scale_grid_divide='add', \
            range_of_div_multiply=(-2, 9), range_of_div_add=(-25, 26), \
            folder_name='Save File', dataset_name='') -> None
        Execute driver for specified scale variations and record timings for each control component.

    Examples
    --------
    >>> TimeAnalysis.analyse(driver, folder_name='timings', dataset_name='set1')
    """

    @staticmethod
    def analyse(driver, scale_grid_space: str = 'multiply', scale_grid_divide = 'add', range_of_div_multiply: Tuple[int, int] = (-2, 9), range_of_div_add: Tuple[int, int] = (-25, 26), folder_name: str = "Save File", dataset_name: str = "") -> None:
        """
        Perform time analysis on driver control functions across grid scaling scenarios.

        .. deprecated:: 2.4.2
            May not function as expected in current versions; slated for revision.

        Parameters
        ----------
        driver : object
            Driver instance with control attributes to profile.
        scale_grid_space : {'multiply', 'add'}, optional
            Method for scaling grid spacing (default: 'multiply').
        scale_grid_divide : {'multiply', 'add'}, optional
            Method for scaling grid divisions (default: 'add').
        range_of_div_multiply : tuple of int, optional
            Range of exponents for multiplicative scaling (default: (-2, 9)).
        range_of_div_add : tuple of int, optional
            Range for additive scaling offsets (default: (-25, 26)).
        folder_name : str, optional
            Directory where timing results will be saved (default: 'Save File').
        dataset_name : str, optional
            Dataset identifier for naming saved results.

        Returns
        -------
        None

        Examples
        --------
        >>> TimeAnalysis.analyse(driver)
        """

        original_grid_space = driver.grid_space
        original_grid_divide = driver.grid_divide

        x_grid_space = []
        x_grid_divide = []

        # Taking multiplicative and additive values for a multiplicative scaled graph and one additively scaled graph based on scale

        if scale_grid_space == 'multiply':

            for i in range(range_of_div_multiply[0], range_of_div_multiply[1]):

                x_grid_space.append(original_grid_space * (1.05 ** i))  # Basis for multiplicative scale is 5 percent of previous value from start and end.

        if scale_grid_space == 'add':

            number = range_of_div_add[1] - range_of_div_add[0] - 1

            space_between = original_grid_space / number

            precision = int(math.ceil(abs(math.log10(space_between))))

            for i in range(range_of_div_add[0], range_of_div_add[1]):

                num = original_grid_space + space_between * i
                x_grid_space.append(round(num, precision))

        if scale_grid_divide == 'add':

            number = range_of_div_add[1] - range_of_div_add[0] - 1

            space_between = int(original_grid_divide / number)

            if space_between == 0:

                space_between = 1

            if space_between * original_grid_divide < abs(range_of_div_add[0]):

                raise RuntimeError(f"Lower Bound too Low for Given Original Divisions\nSolutions\n1. Increase the Lower bound from say -100 to -50\n2. Increase the Original Grid Division Number for Accommodation")


            for i in range(range_of_div_add[0], range_of_div_add[1]):
                num = original_grid_divide + space_between * i
                x_grid_divide.append(num)

        if scale_grid_divide == 'multiply':

            for i in range(range_of_div_multiply[0], range_of_div_multiply[1]):

                x_grid_divide.append(int(original_grid_divide * (2 ** i)))  # Basis for multiplicative scale is 2.0


        original_generate_data_ctrl: bool = driver.generate_data_ctrl  # Used to ctrl whether new data needs to be created.
        original_linear_transform_ctrl: bool = driver.linear_transform_ctrl  # Used to ctrl whether data is linear transformed to 0-1
        original_algorithm_train_ctrl: bool = driver.algorithm_train_ctrl  # Used to ctrl whether to train the algorithm used
        original_calculate_grid_params_ctrl: bool = driver.calculate_grid_params_ctrl  # Used to ctrl whether to calculate grid parameters
        original_compute_grid_ctrl: bool = driver.compute_grid_ctrl  # Used to ctrl whether to compute grid
        original_extract_boundary_ctrl: bool = driver.extract_boundary_ctrl  # Used to ctrl whether to extract boundary
        original_detect_fractal_ctrl: bool = driver.detect_fractal_ctrl  # Used to ctrl whether to detect fractal
        original_calc_connectivity_ctrl: bool = driver.calc_connectivity_ctrl  # Used to ctrl whether to calculate connectivity of boundary
        original_data_display_ctrl: bool = driver.data_display_ctrl  # Used to ctrl whether to display Datapoints only
        original_grid_display_ctrl: bool = driver.grid_display_ctrl  # Used to ctrl whether to display Grid and Grid with Datapoints
        original_display_ctrl: bool = driver.display_ctrl  # Used to ctrl whether to display Boundary, Boundary with Grid, Boundary with Datapoints
        original_force_grid_ctrl: bool = driver.force_grid_ctrl  # Used to ctrl Forced Grid Conversion
        original_display_object_grid_ctrl: bool = driver.display_object_grid_ctrl  # Used to ctrl display of forced grid converted datapoints of given identity
        original_display_hatch_ctrl: bool = driver.display_hatch_ctrl  # Used to ctrl display of hatch of the data point set of given identity
        original_hatch_connectivity_ctrl: bool = driver.hatch_connectivity_ctrl  # Used to ctrl whether to calculate the connectivity factor for hatch
        original_force_grid_connectivity_ctrl: bool = driver.force_grid_connectivity_ctrl  # Use to ctrl whether to calculate connectivity factor for forced grid objects
        original_interpret_ctrl: bool = driver.interpret_ctrl  # Used to ctrl whether to make report or not
        original_save_ctrl: bool = driver.save_ctrl # Used to ctrl whether to save files created in a folder or not (all files must be present refer to the Required_Files.txt to check all required files)
        original_save_force_ctrl: bool = driver.save_force_ctrl  # Used to ctrl saving of force grid files for a given identity
        original_display_time_analysis_ctrl: bool = driver.display_time_analysis_ctrl
        original_save_time_ctrl: bool = driver.save_time_ctrl
        original_clear_ctrl: bool = driver.clear_ctrl
        original_connectivity_deterioration_ctrl: bool = driver.connectivity_deterioration_ctrl
        original_connectivity_deterioration_display_ctrl: bool = driver.connectivity_deterioration_display_ctrl
        original_connectivity_deterioration_save_ctrl: bool = driver.connectivity_deterioration_save_ctrl

        # Should not change to True till end of Time Analysis
        driver.linear_transform_ctrl = False
        driver.connectivity_deterioration_display_ctrl = False
        driver.connectivity_deterioration_save_ctrl = False
        driver.connectivity_deterioration_ctrl = False
        driver.clear_ctrl = False
        driver.save_time_ctrl = False
        driver.time_analysis_ctrl = False
        driver.display_time_analysis_ctrl = False
        driver.save_force_ctrl = False
        driver.save_ctrl = False
        driver.interpret_ctrl = False
        driver.generate_data_ctrl = False
        driver.algorithm_train_ctrl = False
        driver.calculate_grid_params_ctrl = False
        driver.display_ctrl = False
        driver.display_hatch_ctrl = False
        driver.data_display_ctrl = False
        driver.grid_display_ctrl = False
        driver.display_object_grid_ctrl = False

        source_files = [folder_name + f"/({dataset_name})_Datapoints.npy",
                        folder_name + f"/({dataset_name})_Grid_Info.npy",
                        folder_name + f"/({dataset_name})_Grid_Bounds.npy"]  # Replace with the path to your file

        destinations = [os.path.join(os.getcwd(), 'Datapoints.npy'),
                        os.path.join(os.getcwd(), 'Grid_Info.npy'),
                        os.path.join(os.getcwd(), 'Grid_Bounds.npy')]  # Current working directory

        # Copy the file
        for i, source_file in enumerate(source_files):

            if not os.path.isfile(destinations[i]):

                shutil.copy(source_file, destinations[i])


        # Mutable ctrls for analysis
        driver.compute_grid_ctrl = False
        driver.extract_boundary_ctrl = False
        driver.detect_fractal_ctrl = False
        driver.calc_connectivity_ctrl = False
        driver.force_grid_ctrl = False
        driver.hatch_connectivity_ctrl = False
        driver.force_grid_connectivity_ctrl = False

        # Calling Just Once
        driver.algorithm_train_ctrl = True
        driver.main()
        driver.algorithm_train_ctrl = False


        time_grid = []
        time_boundary = []
        time_fractal = []
        time_connectivity = []
        time_force_grid = []
        time_hatch_connectivity = []
        time_force_grid_connectivity = []


        driver.grid_divide = original_grid_divide
        for i in tqdm(range(len(x_grid_space)), desc="Computing Spacing"):

            print("\n")
            driver.grid_space = x_grid_space[i]

            sleep(0.1)

            start_time = time()
            driver.compute_grid_ctrl = True
            driver.main()
            driver.compute_grid_ctrl = False
            time_grid.append(time() - start_time)

            start_time = time()
            driver.extract_boundary_ctrl = True
            driver.main()
            driver.extract_boundary_ctrl = False
            time_boundary.append(time() - start_time)

            start_time = time()
            driver.detect_fractal_ctrl = True
            driver.main()
            driver.detect_fractal_ctrl = False
            time_fractal.append(time() - start_time)

            start_time = time()
            driver.calc_connectivity_ctrl = True
            driver.main()
            driver.calc_connectivity_ctrl = False
            time_connectivity.append(time() - start_time)

            start_time = time()
            driver.force_grid_ctrl = True
            driver.main()
            driver.force_grid_ctrl = False
            time_force_grid.append(time() - start_time)

            start_time = time()
            driver.hatch_connectivity_ctrl = True
            driver.main()
            driver.hatch_connectivity_ctrl = False
            time_hatch_connectivity.append(time() - start_time)

            start_time = time()
            driver.force_grid_connectivity_ctrl = True
            driver.main()
            driver.force_grid_connectivity_ctrl = False
            time_force_grid_connectivity.append(time() - start_time)

            sleep(0.1)


        time_grid_arr = [np.asarray(time_grid.copy())]
        time_boundary_arr = [np.asarray(time_boundary)]
        time_fractal_arr = [np.asarray(time_fractal)]
        time_connectivity_arr = [np.asarray(time_connectivity)]
        time_force_grid_arr = [np.asarray(time_force_grid)]
        time_hatch_connectivity_arr = [np.asarray(time_hatch_connectivity)]
        time_force_grid_connectivity_arr = [np.asarray(time_force_grid_connectivity)]


        time_grid = []
        time_boundary = []
        time_fractal = []
        time_connectivity = []
        time_force_grid = []
        time_hatch_connectivity = []
        time_force_grid_connectivity = []


        driver.grid_space = original_grid_space
        for i in tqdm(range(len(x_grid_divide)), desc="Computing Divisions"):

            driver.grid_divide = x_grid_divide[i]

            print("\n")
            sleep(0.1)

            start_time = time()
            driver.compute_grid_ctrl = True
            driver.main()
            driver.compute_grid_ctrl = False
            time_grid.append(time() - start_time)

            start_time = time()
            driver.extract_boundary_ctrl = True
            driver.main()
            driver.extract_boundary_ctrl = False
            time_boundary.append(time() - start_time)

            start_time = time()
            driver.detect_fractal_ctrl = True
            driver.main()
            driver.detect_fractal_ctrl = False
            time_fractal.append(time() - start_time)

            start_time = time()
            driver.calc_connectivity_ctrl = True
            driver.main()
            driver.calc_connectivity_ctrl = False
            time_connectivity.append(time() - start_time)

            start_time = time()
            driver.force_grid_ctrl = True
            driver.main()
            driver.force_grid_ctrl = False
            time_force_grid.append(time() - start_time)

            start_time = time()
            driver.hatch_connectivity_ctrl = True
            driver.main()
            driver.hatch_connectivity_ctrl = False
            time_hatch_connectivity.append(time() - start_time)

            start_time = time()
            driver.force_grid_connectivity_ctrl = True
            driver.main()
            driver.force_grid_connectivity_ctrl = False
            time_force_grid_connectivity.append(time() - start_time)

            sleep(0.1)


        np.save("X_Grid_Space", np.asarray(x_grid_space))
        np.save("X_Grid_Divide", x_grid_divide)

        time_grid_arr.append(np.asarray(time_grid.copy()))
        np.save("Time_Grid", np.array(time_grid_arr, dtype=object), allow_pickle=True)

        time_boundary_arr.append(np.asarray(time_boundary))
        np.save("Time_Boundary", np.array(time_boundary_arr, dtype=object), allow_pickle=True)

        time_fractal_arr.append(np.asarray(time_fractal))
        np.save("Time_Fractal", np.array(time_fractal_arr, dtype=object), allow_pickle=True)

        time_connectivity_arr.append(np.asarray(time_connectivity))
        np.save("Time_Connectivity", np.array(time_connectivity_arr, dtype=object), allow_pickle=True)

        time_force_grid_arr.append(np.asarray(time_force_grid))
        np.save("Time_Force_Grid", np.array(time_force_grid_arr, dtype=object), allow_pickle=True)

        time_hatch_connectivity_arr.append(np.asarray(time_hatch_connectivity))
        np.save("Time_Hatch_Connectivity", np.array(time_hatch_connectivity_arr, dtype=object), allow_pickle=True)

        time_force_grid_connectivity_arr.append(np.asarray(time_force_grid_connectivity))
        np.save("Time_Force_Grid_Connectivity", np.array(time_force_grid_connectivity_arr, dtype=object), allow_pickle=True)

        # Setting the controls to their original values
        driver.save_force_ctrl = original_save_force_ctrl
        driver.save_ctrl = original_save_ctrl
        driver.interpret_ctrl = original_interpret_ctrl
        driver.generate_data_ctrl = original_generate_data_ctrl
        driver.algorithm_train_ctrl = original_algorithm_train_ctrl
        driver.calculate_grid_params_ctrl = original_calculate_grid_params_ctrl
        driver.display_ctrl = original_display_ctrl
        driver.display_hatch_ctrl = original_display_hatch_ctrl
        driver.data_display_ctrl = original_data_display_ctrl
        driver.grid_display_ctrl = original_grid_display_ctrl
        driver.display_object_grid_ctrl = original_display_object_grid_ctrl
        driver.display_time_analysis_ctrl = original_display_time_analysis_ctrl
        driver.save_time_ctrl = original_save_time_ctrl
        driver.clear_ctrl = original_clear_ctrl
        driver.connectivity_deterioration_ctrl = original_connectivity_deterioration_ctrl
        driver.connectivity_deterioration_display_ctrl = original_connectivity_deterioration_display_ctrl
        driver.connectivity_deterioration_save_ctrl = original_connectivity_deterioration_save_ctrl
        driver.linear_transform_ctrl = original_linear_transform_ctrl


        driver.compute_grid_ctrl = original_compute_grid_ctrl
        driver.extract_boundary_ctrl = original_extract_boundary_ctrl
        driver.detect_fractal_ctrl = original_detect_fractal_ctrl
        driver.calc_connectivity_ctrl = original_calc_connectivity_ctrl
        driver.force_grid_ctrl = original_force_grid_ctrl
        driver.hatch_connectivity_ctrl = original_hatch_connectivity_ctrl
        driver.force_grid_connectivity_ctrl = original_force_grid_connectivity_ctrl

        return None
