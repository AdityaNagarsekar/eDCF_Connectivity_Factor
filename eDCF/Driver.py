# import statements
import numpy as np
from typing import List

from Connectivity import Connectivity
from ConnectivityDeterioration import ConnectivityDeterioration
from ForceGridObject import ForceGridObject
from Manager import Manager
from TimeAnalysis import TimeAnalysis
from algorithms import *
from DynamicDataGenerator import DynamicDataGenerator
from data_structures import *
from Interpreter import Interpreter
from Save import Save
from TopologicalDimension import TopologicalDimension
from FractalDetector import FractalDetector


class Driver:
    """

    The class Driver is used to control the work flow of the entire system to ensure smooth operation of different components.

    Contains:
    -----

    1. __init__() - Default Constructor for the class which is used to set the parameters for testing various types of data.
    2. main() - main function of the Driver which is intended to execute the script and handle the work flow based on the __init__ values.

    """

    def __init__(self):
        """
        Used to define the parameters to run the System

        Tips(from version v_2.0.1):
        -----

        1. During Testing take values from DatapointsStorage Set{_} instead of generating a new set strictly i.e. generate_data_ctrl = False always for Testing.
        2. Do not put save_ctrl = True unless sure about the display format and colours and set everything else to false while saving.
        3. Follow the TestingTemplates strictly for testing
        4. For taking values for sets use ctrl+c and NEVER use ctrl+x
        5. Identity should always be strictly greater than 0

        NOTE (from version 2.3.1)
        -----

        1. The method of NCubeNeighbours, while is heavy in terms of RAM and the computation of higher dimensional neighbours provides a much better time than the method of Neighbours in Set since it uses set intersection which uses hash lookups instead of looping conditions.

        2. It is important to note that NCubeNeighbours struggles with higher dimensional data both in terms of computation speed and spacial complexity but is better than Neighbour in Set method for higher data volume.

        3. It is however necessary to use Neighbour in Set method for higher dimensional data even if it is very slow.

        4. Advancements can most likely be made.

        Happy Testing to whoever is testing :)
        """

        # CTRL VARIABLES

        # ----------

        self.core_ctrl: int = 10  # the learning algorithms are crashing on n_jobs = -1 so we are controlling it through this.

        # PART 1 - Generating the Data for Testing (uses data_structures package)
        # The existence of Datapoints.npy is necessary for the execution of the script.

        self.generate_data_ctrl: bool = False  # Used to ctrl whether new data needs to be created.
        self.linear_transform_ctrl: bool = False  # Used to ctrl whether to linear transform the data to 0 - 1 or keep it as it is. We have used Global Normalization.

        # PART 2 - Algorithm Training (uses algorithms package)

        self.algorithm_train_ctrl: bool = False  # Used to ctrl whether to train the algorithm used

        # PART 3 - Calculating Grid Information (uses the Manager.py file)

        self.calculate_grid_params_ctrl: bool = False  # Used to ctrl whether to calculate grid parameters

        # PART 4 - Generating Grid, Extracting Boundary from Grid, Detecting Box Counting Dimension, Calculating Connectivity Factor for Boundary
        # Uses GridGenerator.py, BoundaryExtractor.py, FractalDetector.py, Connectivity.py files respectively.

        self.compute_grid_ctrl: bool = False  # Used to ctrl whether to compute grid
        self.extract_boundary_ctrl: bool = False  # Used to ctrl whether to extract boundary
        self.detect_fractal_ctrl: bool = False  # Used to ctrl whether to detect fractal dimension of boundary
        self.calc_connectivity_ctrl: bool = False  # Used to ctrl whether to calculate connectivity of boundary

        # PART 5 - Displaying the Different Graphs using the Data computed in the previous PARTS (uses the Manager.py file)

        self.data_display_ctrl: bool = False  # Used to ctrl whether to display Datapoints only
        self.grid_display_ctrl: bool = False # Used to ctrl whether to display Grid and Grid with Datapoints
        self.display_ctrl: bool = False  # Used to ctrl whether to display Boundary, Boundary with Grid, Boundary with Datapoints

        # PART 6 - Forcing Objects to be Grid Data Points instead of arbitrary floating point values (uses ForceGridObject.py file)

        self.force_grid_ctrl: bool = False # Used to ctrl Forced Grid Conversion
        self.direct_conversion_ctrl: bool = False # Used to ctrl whether to use direct conversion or not
        self.range_analysis_ctrl: bool = False # Used to ctrl whether to perform this operation. When turned on if step is -ve it does not perform as expected.
        self.range_display_ctrl: bool = False # Used to ctrl whether to display the range analysis graphs

        # NOTE: We do not get Hatch if direct_conversion is used.

        # PART 7 - Displaying the Different Graphs using Data Computed in PART 6 (uses Manager.py file)

        self.display_object_grid_ctrl: bool = False  # Used to ctrl display of forced grid converted datapoints of given identity
        self.display_hatch_ctrl: bool = False  # Used to ctrl display of hatch of the data point set of given identity

        # PART 8 - Calculating Connectivity of Hatch of Object and Forced Grid Points of the Object (uses Connectivity.py file)

        self.hatch_connectivity_ctrl: bool = False  # Used to ctrl whether to calculate the connectivity factor for hatch
        self.force_grid_connectivity_ctrl: bool = False  # Use to ctrl whether to calculate connectivity factor for forced grid objects

        # PART 9 - Topological Dimension Estimation based on Connectivity Factors (Uses Connectivity.py and NCubeNeighbours.py files)
        self.topological_dimension_ctrl: bool = False  # Used to ctrl whether to calculate the connectivity factor of hatches, force grid objects and/or boundaries.
        self.fractal_object_ctrl: bool = False

        # PART 10 - Interpreting the Data in the Previous Parts
        # Uses Interpreter.py

        self.interpret_ctrl: bool = False  # Used to ctrl whether to make report or not

        # PART 11 - Connectivity Deterioration for an Object with time for random points ejection. (uses ConnectivityDeterioration.py file)
        # Additional use file - Manager.py

        self.connectivity_deterioration_ctrl: bool = False  # Used to ctrl whether connectivity deterioration of an object is computed
        self.connectivity_deterioration_display_ctrl: bool = False  # Used to display connectivity deterioration graphs of object

        # PART 12 - Saving the Data to Specified <Folder>/<Sub Folder> (uses Save.py file)

        self.save_force_ctrl: bool = False  # Used to ctrl saving of force grid files for a given identity
        self.connectivity_deterioration_save_ctrl: bool = False  # Used to save the results of connectivity deterioration for an object list.
        self.save_ctrl: bool = False  # Used to ctrl whether to save files created in a folder or not (all files must be present refer to the Required_Files.txt to check all required files)

        # PART 13 - Time Analysis for the System (uses TimeAnalysis.py file)
        # Additional use files - Manager.py and Save.py

        self.time_analysis_ctrl: bool = False  # Be Careful this may take a long time to perform.
        self.display_time_analysis_ctrl: bool = False  # Used to ctrl display of the graphs of time analysis.
        self.save_time_ctrl: bool = False  # Used to ctrl saving of time analysis related files

        # PART 14 - Clearing of the System for fresh use (uses Save.py file)
        # CAUTION: Please Save before use of this as this can lead to Data Loss if not used properly.

        self.clear_ctrl: bool = False  # Clears Files instead of saving them to clean the system for operation.

        # ----------

        # DynamicDataGenerator parameters to set if generate_data_ctrl is True

        # ----------

        # Used to set the data structures that need to be generated

        self.data_objects = [Circle.Circle(identity=1, radius=3.0, center=(0.0, 0.0), noise_rate=0.0), Circle.Circle(identity=2, radius=4.0, center=(0.0, 0.0), noise_rate=0.1)]
        self.data_objects_names = ['Circle(identity=1, radius=3.0, center=(0.0, 0.0), noise_rate=0.0)', 'Circle(identity=2, radius=4.0, center=(0.0, 0.0), noise_rate=0.1)']


        self.num_points: List[int] | int = 15000  # used to set the number of points in each structure.

        self.dataset_name = 'CCD_DecisionTree'

        # ----------

        # Algorithm to call if algorithm_train_ctrl is True (please see Algorithms_Template.txt for use)

        # ----------

        self.algorithm = DecisionTree.DecisionTree(criterion=['gini', 'entropy'], splitter=['best', 'random'],
                                                   max_depth=[5, 10, 20, 50, 100], min_samples_split=[2, 5, 10],
                                                   min_samples_leaf=[1, 2, 4], max_features=[None, 'sqrt', 'log2'])
        self.algorithm_name = 'DecisionTree(criterion=[\'gini\', \'entropy\'], splitter=[\'best\', \'random\'], max_depth=[5, 10, 20, 50, 100], min_samples_split=[2, 5, 10], min_samples_leaf=[1, 2, 4], max_features=[None, \'sqrt\', \'log2\'])'

        # ----------

        # Project Fractal Parameters if any of compute_grid_ctrl, extract_boundary_ctrl or detect_fractal_ctrl is True to make the ProjectFractal Object

        # ----------

        self.grid_space: float = 0.0005  # grid spacing between each point in the grid
        self.bounds_buffer: float = 0.0  # extension provided to the grid to prevent losses (assigning 0.0 makes it so that we get automatic least value to prevent losses)
        self.grid_divide: int = 50  # for division of grids into mini grids for multiprocessing. This value represents the division each axis will go through in the grid
        self.neighbour_set_method: List[bool] = [False, False]  # it is a list containing whether to use the neighbour set method or not. 0 index is for boundary extraction, 1 index is for connectivity factor.

        # ----------

        # Directory Name to Save to if save_ctrl is True

        # ----------

        self.directory_name = "Data/Random_Experiment_7"  # file path followed by the folder name you want to store the data in

        # ----------

        # When force_grid_ctrl is True the label list to force grid

        # ----------

        self.force_identity: List[int] = [1, 2]  # List of objects that are chosen for Force Grid Conversion and also for Connectivity Deterioration

        # ----------

        # When Time Analysis Ctrl is True

        # ----------

        self.scale_grid_space = 'multiply'  # choosing whether we want the scale of spacing to be 'multiply' based or 'add' based.
        self.scale_grid_divide = 'add'  # choosing whether we want the scale of grid divisions to be 'multiply' or 'add' based.
        self.range_of_div_multiply = (-100, 101)  # choosing the range we want for a scale of 'multiply' type.
        self.range_of_div_add = (-100, 101)  # choosing the range we want for a scale of 'add' type.

        # ----------

        # When connectivity_deterioration_ctrl is True

        # ----------

        self.x_points_num = 2500  # No. of points on the x-axis(time/length(total number) points)
        self.delete_points_max = 5  # maximum number of points allowed to delete per mini grid per iteration.
        self.probability_deter = 0.2  # probability of a mini grid being selected for deletion of points.

        # ----------

        # SPACE ALLOCATOR VARS

        # ----------

        # Change according to requirements
        # We will take percentage of the number of force grid points compared to the total number of points that are present in the data for that object.
        self.lower_percent_cap: float = 0
        self.upper_percent_cap: float = 100
        self.step: float = 2.5  # set step = -1.0 if we have original force grid conversion method with hatch otherwise you can use whichever valid value you want.
        self.display_lower: float = 50.0
        self.basis: float = 50.0

        # Take care in assigning percentage as it is possible that the values are generated such that the percentage performs an unexpected jump at the difference of more than possible to calculate precision.
        # Therefore, please do not assign small ranges like 98 to 99 since we may not get there ever resulting in a segmentation fault.

        # (try not to change)
        self.degree: int = 5
        self.start_checker: float = 0.1
        self.end_checker: float = 0.0000001
        self.binary_iteration_limit: int = 50  # to prevent segmentation fault the max number of binary searches available.
        self.changing_factor: int = 5
        self.bias: float = 0.95

        # Always run calculate_grid_params_ctrl in case of crash (VVVIMP)
        self.dynamic_spacing_ctrl: bool = True  # if True then the spacing at which the object is force grided is not the same as the one provided above and is instead dynamically allocated based on the dataset.

        # ----------

        # DO NOT CHANGE THESE LINES

        # ----------

        self.manager = ...
        self.interpret = ...
        self.save = ...

        # ----------

    def main(self) -> None:
        """
        This function is the main driver function which needs to be called for execution of the script and thus the entire system.

        :return: None
        """

        # PART 1 Execution

        # ----------

        if self.generate_data_ctrl:

            print("\nGenerating Data... {CAUTION: WILL OVERWRITE DATA IN Datapoints.npy}")
            if self.linear_transform_ctrl:
                DynamicDataGenerator(self.data_objects, self.num_points).generate_data(linear_transform=True)
            else:
                DynamicDataGenerator(self.data_objects, self.num_points).generate_data()
            print("\nNew Data Generated Successfully.")

        # ----------

        # Making a Manger object for further use.

        self.manager = Manager(algorithm=self.algorithm, grid_space=self.grid_space, bounds_buffer=self.bounds_buffer, grid_divide=self.grid_divide, dataset_name=self.dataset_name)

        # PART 2 Execution

        # ----------

        if self.algorithm_train_ctrl:

            print("\nAlgorithm Training in Progress...")
            self.algorithm.train_hyper(self.core_ctrl)
            print("\nTraining Completed Successfully.")

        # ----------

        # PART 3 Execution

        # ----------

        if self.calculate_grid_params_ctrl:

            print("\nCalculating Grid Parameters...")
            self.manager.calculate_grid_information()
            print("\nGrid Parameters Calculated Successfully.")

        # ----------

        # PART 4 Execution

        # ----------

        if self.compute_grid_ctrl:

            print("\nComputing Divided Grid...")
            self.manager.compute_grid()
            print("\nGrid Computed Successfully.")

        if self.extract_boundary_ctrl:

            print("\nExtracting Boundary...")
            self.manager.compute_boundary_neighbour(weight=0.0, neighbour_set_method=self.neighbour_set_method[0])
            print("\nBoundary Extracted Successfully.")

        if self.detect_fractal_ctrl:

            print("\nDetecting Fractal Dimension...")
            self.manager.fractal_detection()
            print("\nDimension Calculated Successfully.")

        if self.calc_connectivity_ctrl:

            print("\nCalculating Connectivity Factors for Boundary System...")
            Connectivity.calc_connectivity(self.grid_space, neighbour_set_method=self.neighbour_set_method[1])
            print("\nConnectivity Factors Calculated Successfully.")

        # ----------

        # PART 5 Execution

        # ----------

        if self.data_display_ctrl:

            print("\nDisplaying Datapoints...")
            self.manager.display_data()
            print("\nDatapoints displayed in file Data_Points_Plot.png.")

        if self.grid_display_ctrl:

            print("\nDisplaying Grid...")
            self.manager.display_grid()
            print("\nGrid displayed in 2 formats in files Grid_Plot.png and Grid_Plot_Datapoints.png.")

        if self.display_ctrl:

            print("\nDisplaying Boundary...")
            self.manager.display()
            print("\nGrid displayed in 3 formats in files Boundary_Plot.png and Grid_Plot_Boundary.png and Data_Points_Plot_Boundary.png.")

        # ----------

        if self.step == -1.0 or not self.range_analysis_ctrl:

            # PART 6 Execution

            # ----------

            if self.force_grid_ctrl:

                np.save("Spacing_Allocation_Details.npy", np.array([self.start_checker, self.end_checker, self.lower_percent_cap, self.upper_percent_cap, self.binary_iteration_limit]))
                print("\nForce Grid Conversion...")
                for identity in self.force_identity:
                    print(f"\nObject {identity} Conversion...")
                    if not ForceGridObject(identity).compute(self.direct_conversion_ctrl, self.dynamic_spacing_ctrl, changing_factor=self.changing_factor):
                        print("\nCalculating Grid Parameters...")
                        self.manager.calculate_grid_information()
                        print("\nGrid Parameters Calculated Successfully.")
                        print(f"\nObject {identity} Recalculation Attempt...")
                        ForceGridObject(identity).compute(self.direct_conversion_ctrl, self.dynamic_spacing_ctrl, changing_factor=self.changing_factor)
                    print(f"\nObject {identity} Conversion Complete.")
                print("\nForced Grid Computation Complete.")
                spacings = []
                info = np.load("Grid_Info.npy")
                for i in range(len(self.force_identity)):
                    spacings.append(info[i + 4])
                info = np.delete(info, np.s_[-len(self.force_identity):])  # there might be a warning because I added the colon, but it is correct and works as intended.
                np.save("Grid_Info.npy", info)
                np.save("Force_Grid_Spacings.npy", np.array(spacings))

            # ----------

            # PART 7 Execution

            # ----------

            if self.display_object_grid_ctrl:
                print("\nDisplaying Grid...")
                for identity in self.force_identity:
                    print(f"\nDisplaying Object {identity}...")
                    self.manager.display_object_grid(identity, direct_computation_ctrl=self.direct_conversion_ctrl)
                    print(f"\nDisplayed Object {identity} in files Grid_Plot_Object_{identity}.png and Grid_Plot_Datapoints_Object_{identity}.png.")
                print("\nGrid displayed.")

            if self.display_hatch_ctrl:

                print("\nDisplaying Hatch...")
                for identity in self.force_identity:
                    print(f"\nDisplaying Hatch Object {identity}...")
                    self.manager.display_hatch(identity)
                    print(f"\nHatch Object {identity} Displayed in Hatch_Object_{identity}.png.")
                print(f"\nHatch displayed.")

            # ----------

            # PART 8 Execution

            # ----------

            if self.hatch_connectivity_ctrl:

                cfs = []
                spacings = np.load("Force_Grid_Spacings.npy")
                weights_hatch = []
                print("\nCalculating Connectivity of Hatch...")
                for i, identity in enumerate(self.force_identity):
                    print(f"\nObject {identity} Computation...")
                    intermediate = Connectivity.calc_connectivity_general(np.load(f"Hatch_Object_{identity}.npy"), spacings[i], neighbour_set_method=self.neighbour_set_method[1])

                    if intermediate == -1:
                        intermediate = (-1.0, Weight(int(np.load("Grid_Info.npy")[3])))

                    cfs.append(intermediate[0])
                    weights_hatch.append(intermediate[1].weights)
                    print(f"\nObject {identity} Computation Complete.")
                print("\nConnectivity Computation Complete.")

                weights_hatch = np.array(weights_hatch)

                np.save("Hatch_Connectivity_Factors", cfs)
                np.save("Weights_Hatch.npy", weights_hatch)

            if self.force_grid_connectivity_ctrl:

                cfs = []
                weights_force = []
                spacings = np.load("Force_Grid_Spacings.npy")
                print("\nCalculating Connectivity of Forced Grid Data...")
                for i, identity in enumerate(self.force_identity):
                    print(f"\nObject {identity} Computation...")
                    grid_data = []
                    grid_points = np.load(f"Grid_Points_Object_{identity}.npy")
                    try:
                        for grid in grid_points:
                            grid_data.append(grid[grid[:, -1] == identity])
                    except IndexError:
                        grid_data.append(grid_points)

                    intermediate = Connectivity.calc_connectivity_general(grid_data, spacings[i], neighbour_set_method=self.neighbour_set_method[1])

                    if intermediate == -1:
                        intermediate = (-1.0, Weight(int(np.load("Grid_Info.npy")[3])))
                    
                    cfs.append(intermediate[0])
                    weights_force.append(intermediate[1].weights)
                    print(f"\nObject {identity} Computation Complete.")
                print("\nConnectivity Computation Complete.")

                weights_force = np.array(weights_force)

                np.save("Object_Connectivity_Factors", cfs)
                np.save("Weights_Force.npy", weights_force)

            # ----------

            if self.range_display_ctrl:

                for i in range(len(self.force_identity)):
                    print(f"\nPlotting CF vs Percentage Graph for Object - {self.force_identity[i]}...")
                    self.manager.analysis_object_display(i, self.force_identity)
                    print(f"\nPlotting Complete for Object - {self.force_identity[i]}")

                    print(f"\nPlotting Weights vs Percentage Graph for Object - {self.force_identity[i]}...")
                    self.manager.weights_object_display(i, self.force_identity)
                    print(f"\nPlotting Complete for Object - {self.force_identity[i]}")

        elif self.step > 0.0 and self.range_analysis_ctrl:

            analysis = np.array([])
            weights_analysis = np.asarray([])
            lower_cap = self.lower_percent_cap
            higher_cap = self.step + self.lower_percent_cap
            np.save("Analysis", analysis)
            np.save("Weights_Analysis.npy", weights_analysis)

            while higher_cap <= self.upper_percent_cap:

                print(f"\nComputation of {lower_cap} - {higher_cap}...")

                if self.calculate_grid_params_ctrl:
                    print("\nCalculating Grid Parameters...")
                    self.manager.calculate_grid_information()
                    print("\nGrid Parameters Calculated Successfully.")

                # PART 6 Execution

                # ----------

                if self.force_grid_ctrl:

                    np.save("Spacing_Allocation_Details.npy", np.array(
                        [self.start_checker, self.end_checker, lower_cap, higher_cap,
                         self.binary_iteration_limit]))
                    print("\nForce Grid Conversion...")
                    for identity in self.force_identity:
                        print(f"\nObject {identity} Conversion...")
                        if not ForceGridObject(identity).compute(self.direct_conversion_ctrl, self.dynamic_spacing_ctrl,
                                                                 changing_factor=self.changing_factor):
                            print("\nCalculating Grid Parameters...")
                            self.manager.calculate_grid_information()
                            print("\nGrid Parameters Calculated Successfully.")
                            print(f"\nObject {identity} Recalculation Attempt...")
                            ForceGridObject(identity).compute(self.direct_conversion_ctrl, self.dynamic_spacing_ctrl,
                                                              changing_factor=self.changing_factor)
                        print(f"\nObject {identity} Conversion Complete.")
                    print("\nForced Grid Computation Complete.")
                    spacings = []
                    info = np.load("Grid_Info.npy")
                    for i in range(len(self.force_identity)):
                        spacings.append(info[i + 4])
                    info = np.delete(info, np.s_[-len(self.force_identity):])  # there might be a warning because I added the colon, but it is correct and works as intended.
                    np.save("Grid_Info.npy", info)
                    np.save("Force_Grid_Spacings.npy", np.array(spacings))

                # ----------

                percent_range = [lower_cap, higher_cap]
                spacings_objects = np.load("Force_Grid_Spacings.npy")
                analysis = list(np.load("Analysis.npy"))
                weights_analysis = list(np.load("Weights_Analysis.npy"))
                analysis_entry = []

                for i in range(len(self.force_identity)):

                    percent_range.append(spacings_objects[i])
                    analysis_entry.append(percent_range.copy())
                    percent_range.pop(-1)
                    force_grid = np.load(f"Grid_Points_Object_{self.force_identity[i]}.npy")
                    np.save(f"Grid_Points_Object_{self.force_identity[i]}_{lower_cap}-{higher_cap}.npy", force_grid)

                if self.force_grid_connectivity_ctrl:

                    cfs = []
                    weights_force = []
                    spacings = np.load("Force_Grid_Spacings.npy")
                    print("\nCalculating Connectivity of Forced Grid Data...")
                    for i, identity in enumerate(self.force_identity):
                        print(f"\nObject {identity} Computation...")
                        grid_data = []
                        grid_points = np.load(f"Grid_Points_Object_{identity}.npy")
                        try:
                            for grid in grid_points:
                                grid_data.append(grid[grid[:, -1] == identity])
                        except IndexError:
                            grid_data.append(grid_points)

                        intermediate = Connectivity.calc_connectivity_general(grid_data, spacings[i], neighbour_set_method=
                        self.neighbour_set_method[1])

                        if intermediate == -1:
                            intermediate = (-1.0, Weight(int(np.load("Grid_Info.npy")[3])))

                        cfs.append(intermediate[0])
                        weights_force.append(intermediate[1].weights)

                        print(f"\nObject {identity} Computation Complete.")
                    print("\nConnectivity Computation Complete.")

                    weights_force = np.array(weights_force)

                    np.save("Object_Connectivity_Factors", cfs)
                    np.save("Weights_Force.npy", weights_force)


                try:
                    object_cfs = np.load("Object_Connectivity_Factors.npy")
                    object_weights = np.load("Weights_Force.npy")

                    weights_analysis.append(object_weights)

                    for i in range(len(self.force_identity)):

                        percent_range = analysis_entry[i]
                        percent_range.append(object_cfs[i])
                        percent_range.append(np.load(f"Percent_{self.force_identity[i]}.npy")[0])

                except FileNotFoundError:
                    pass

                analysis.append(analysis_entry)

                np.save("Analysis", np.array(analysis))
                np.save("Weights_Analysis.npy", np.array(weights_analysis))

                print(f"\nComputation of {lower_cap} - {higher_cap} complete.")

                lower_cap = higher_cap
                higher_cap = lower_cap + self.step

            analysis = np.load("Analysis.npy")
            weights_analysis = np.load("Weights_Analysis.npy")

            # We’ll collect the filtered results in a list (one entry per `i`)
            filtered_results = []
            final_wts = []

            for i in range(analysis.shape[1]):
                # Extract columns for this "i"
                a = analysis[:, i, 0]
                b = analysis[:, i, 1]
                x = analysis[:, i, 4]
                y = analysis[:, i, 3]
                z = analysis[:, i, 2]  # example: you mentioned taking log of col 2

                wt_i = weights_analysis[:, i]

                # Outlier removal using a while-loop
                j = 0
                while j < len(y):

                    # if j == 0:
                    #     j += 1
                    #     continue

                    # Your condition
                    # if ((abs(y[j] - y[j - 1]) > self.bias or
                    #      abs(y[j + 1] - y[j]) > self.bias)
                    #         and (y[j] - y[j - 1] < 0 and y[j + 1] - y[j] > 0)):
                    
                    weight_iter = iter(wt_i[j].items())

                    wt_i_arr = []
                    sum_weights = 0

                    for key, value in weight_iter:
                        sum_weights += value
                        wt_i_arr.append(value)
                    
                    wt_i_arr = np.array(wt_i_arr) / sum_weights

                    if(wt_i_arr[0] > self.bias):

                        a = np.delete(a, j)
                        b = np.delete(b, j)
                        x = np.delete(x, j)
                        y = np.delete(y, j)
                        z = np.delete(z, j)

                        wt_i = np.delete(wt_i, j)

                        j -= 1
                    else:
                        j += 1

                # Remove first and last points by slicing
                # a = a[1:-1]
                # b = b[1:-1]
                # x = x[1:-1]
                # y = y[1:-1]
                # z = z[1:-1]

                # wt_i = wt_i[1:-1]

                # Store the cleaned data for this "i"
                # Each "i" may end up with a different length after outlier removal.
                # We can store them as tuples or arrays of different lengths in a list:
                filtered_temp = []
                filtered_weights = []

                for k in range(len(y)):
                    filtered_temp.append([a[k], b[k], z[k], y[k], x[k]])
                    filtered_weights.append(wt_i[k])
                filtered_results.append(filtered_temp)
                final_wts.append(filtered_weights)

            np.save("Analysis.npy", np.array(filtered_results, dtype=object))
            np.save("Weights_Analysis.npy", np.array(final_wts, dtype=object))

            if self.range_display_ctrl:

                for i in range(len(self.force_identity)):
                    print(f"\nPlotting CF vs Percentage Graph for Object - {self.force_identity[i]}...")
                    self.manager.analysis_object_display(i, self.force_identity)
                    print(f"\nPlotting Complete for Object - {self.force_identity[i]}")

                    print(f"\nPlotting Weights vs Percentage Graph for Object - {self.force_identity[i]}...")
                    self.manager.weights_object_display(i, self.force_identity)
                    print(f"\nPlotting Complete for Object - {self.force_identity[i]}")

            if self.display_object_grid_ctrl:
                print("\nDisplaying Grid...")
                for identity in self.force_identity:
                    print(f"\nDisplaying Object {identity}...")
                    self.manager.display_object_grid(identity, direct_computation_ctrl=self.direct_conversion_ctrl, perc_range_lower=self.display_lower, perc_range_upper=self.display_lower + self.step)
                    print(f"\nDisplayed Object {identity} in files Grid_Plot_Object_{identity}.png and Grid_Plot_Datapoints_Object_{identity}.png.")
                print("\nGrid displayed.")

        else:

            raise ValueError("Step Value can be -1.0 or Strictly Greater than 0.0")

        # PART 9 Execution

        # ----------

        if self.topological_dimension_ctrl:
            print("\nEstimating Topological Dimensions...")
            TopologicalDimension().execute(self.basis, self.degree)
            print("\nFinished assessing Topological Dimensions and they are stored in Topological_Dimensions_Boundaries.npy, Topological_Dimensions_Object.npy and Topological_Dimensions_Hatch.npy")

        # ----------

        if self.fractal_object_ctrl:
            frac_d = FractalDetector()
            datapoints_list = np.load("Datapoints.npy", allow_pickle=True)
            fractal_dims = []

            print("\nEstimating Fractal Dimensions...")
            for identity in self.force_identity:
                print(f"\nFor Object {identity}...")
                fractal_dims.append(frac_d.detect_general(datapoints_list[identity - 1]))
                print(f"\nCompleted for Object {identity}.")
            print("\nEstimation Complete.")

            np.save("Fractal_Dimensions_Objects.npy", fractal_dims)

        # PART 10 Execution

        # ----------

        if self.interpret_ctrl:

            self.interpret = Interpreter(structures=self.data_objects_names, algorithm=self.algorithm_name)
            
            print("\nInterpreting to write Report...")
            try:
                self.interpret.report(self.neighbour_set_method)
                print("\nReport Writing Successful, available in Report.txt.")
            except FileNotFoundError:
                print("\n1 or more Required Files are missing.")
            try:
                self.interpret.add_force_info(self.force_identity, self.basis, self.dynamic_spacing_ctrl)
            except FileNotFoundError:
                pass
            self.interpret.add_topological_info(self.force_identity)
            self.interpret.add_fractal_obj_info(self.force_identity)

        # ----------

        # PART 11 Execution

        # ----------

        if self.connectivity_deterioration_ctrl:

            print("\nComputing Connectivity Deterioration with Time...")
            for identity in self.force_identity:
                print(f"\nObject {identity} Computation...")
                ConnectivityDeterioration(identity=identity, obj=np.load(f"Grid_Points_Object_{identity}.npy"),
                                          x_points_num=self.x_points_num,
                                          delete_points_max=self.delete_points_max,
                                          probability_deter=self.probability_deter).connectivity_deterioration_analysis()
                print(f"\nObject Computation Complete.")
            print("\nTask Completed Successfully.")

        if self.connectivity_deterioration_display_ctrl:

            print("\nDisplaying Connectivity Deterioration...")
            for identity in self.force_identity:
                print(f"\nDisplaying Deterioration of Object {identity}...")
                self.manager.connectivity_deterioration_display(identity)
                print(f"\nDeterioration of Object {identity} Displayed in Time_Length_Object_{identity}.png.")
            print(f"\nDeterioration displayed.")

        # ----------

        # PART 12 Execution

        # ----------

        if(self.save or self.save_force_ctrl or self.save_time_ctrl or self.connectivity_deterioration_save_ctrl or self.clear_ctrl):
            self.save = Save(dataset_name=self.dataset_name,directory_name=self.directory_name)


        if self.save_force_ctrl:

            for identity in self.force_identity:
                print(f"\nSaving to Folder {self.directory_name}/Force_Grid_Object_{identity}...")
                self.save.save_force(identity)
                self.save.range_force_save(identity, self.lower_percent_cap, self.upper_percent_cap, self.step)
                self.save.cleanup(self.force_identity)
                print("\nSaving Successful.")

        if self.connectivity_deterioration_save_ctrl:

            for identity in self.force_identity:
                print(f"\nSaving to Folder {self.directory_name}/Deterioration_Object_{identity}...")
                self.save.connectivity_deter_save(identity)
                print("\nSaving Successful.")


        if self.save_ctrl:

            print(f"\nSaving to Folder {self.directory_name}...")
            self.save.save()
            print("\nSaving Successful.")

            try:
                self.save.save_force_connectivity()
            except FileNotFoundError:
                pass

        # ----------

        # PART 13 Execution

        # ----------

        if self.time_analysis_ctrl:

            print("\nStarting Time Analysis...{Please have Patience this might take a while}")

            TimeAnalysis.analyse(self, scale_grid_space=self.scale_grid_space, scale_grid_divide=self.scale_grid_divide,
                                 range_of_div_multiply=self.range_of_div_multiply,
                                 range_of_div_add=self.range_of_div_add, folder_name=self.directory_name,
                                 dataset_name=self.dataset_name)

            print("\nTime Analysis Complete.")

        if self.display_time_analysis_ctrl:

            print("\nDisplaying Time Analysis...")
            self.manager.display_time_analysis()
            print("\nDisplayed Time Analysis.")

        if self.save_time_ctrl:

            print(f"\nSaving Time Analysis files to folder {self.directory_name}/Time_Analysis...")
            self.save.save_time_analysis()
            print("\nSaving Successful.")

        # ----------

        # PART 14 Execution

        # ----------

        if self.clear_ctrl:

            print("\nClearing System...")
            self.save.save(clear=True)
            for identity in self.force_identity:
                self.save.save_force(identity=identity, clear=True)
            self.save.save_force_connectivity(clear=True)
            self.save.save_time_analysis(clear=True)
            for identity in self.force_identity:
                self.save.connectivity_deter_save(identity=identity, clear=True)
                self.save.range_force_save(identity, self.lower_percent_cap, self.upper_percent_cap, self.step, clear=True)
            self.save.cleanup(self.force_identity)
            print("\nClearing Complete.")

        # ----------

# VVVV (Starting the Script from Below) VVVV

if __name__ == '__main__':

    Driver().main()
