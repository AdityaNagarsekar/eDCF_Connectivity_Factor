# import statements
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import colorsys

from BoundaryExtractor import BoundaryExtractor
from FractalDetector import FractalDetector
from GridGenerator import GridGenerator

class Manager:
    """
    Central Hub for Data Processing and Visualization

    Orchestrates the end-to-end pipeline: grid calculation, boundary extraction,
    fractal detection, connectivity analysis, and result visualization.

    Parameters:
        algorithm (Algorithm)        : grid generation & analysis algorithm
        grid_space (float)           : spacing between grid points
        bounds_buffer (float)        : extra margin on axis boundaries
        grid_divide (int)            : number of divisions per axis
        dataset_name (str)           : identifier for output filenames

    Output Files:
        Grid_Info.npy                      : [spacing, buffer, divisions, dimension]
        Grid_Bounds.npy                    : axis-aligned grid boundaries
        Datapoints.npy                     : raw data points array
        Data_Points_Plot.png               : plot of raw data points
        Grid_Plot.png                      : plot of grid structure
        Boundary_Plot.png                  : plot of extracted boundary
        Data_Points_Plot_Boundary.png      : data & boundary overlay
        Grid_Plot_Boundary.png             : grid & boundary overlay
        CF_Percentage_Object_{id}.png      : connectivity factor % per object
        CF_Spacing_Object_{id}.png         : connectivity vs ln(spacing)
        Weights_Percentage_Object_{id}.png : weight distribution % per object
        HATCH_Object_{id}.png              : hatch visualization per object
        Time_*.png                         : performance metric graphs

    Public Methods:
        Computation:
            warning_bounds()

            calculate_grid_information()

            compute_grid()

            compute_boundary_neighbour(weight, neighbour_set_method)

            fractal_detection()
            
            plot_x_y_time(x, y, name_x, type_graph)
            
            plot_weights_vs_percentage(weights_perc_norm, x, title, xlabel, ylabel, object_id)

        Visualization:
            display_data()

            display_grid()

            display()

            display_object_grid(identity, direct_computation_ctrl, perc_range_lower, perc_range_upper)

            display_hatch(identity)

            display_time_analysis()

            connectivity_deterioration_display(identity)

            analysis_object_display(index, force_identity)

            weights_object_display(index, force_identity)

    Usage Example:
        >>> manager = Manager(my_algo, 0.1, 0.5, 10, "MyDataset")
        >>> manager.calculate_grid_information()
        >>> manager.compute_grid()
        >>> manager.compute_boundary_neighbour()
        >>> manager.fractal_detection()
        >>> manager.display()

    Note:
        All output files are saved as .npy or .png in the current directory.
    """

    def __init__(self, algorithm, grid_space: float, bounds_buffer: float, grid_divide: int, dataset_name: str):
        """
        Initialize the Manager with grid configuration and dataset settings.

        Parameters:
            algorithm (Algorithm): algorithm for grid generation and analysis
            grid_space (float): spacing between adjacent grid points
            bounds_buffer (float): margin added to axis boundaries
            grid_divide (int): number of divisions per grid axis
            dataset_name (str): identifier prefix for output files
        """

        self.algorithm = algorithm
        self.grid_space: float = grid_space
        self.bounds_buffer: float = bounds_buffer  # boundary of the entire grid has a buffer value
        self.grid_divide: int = grid_divide  # division of grid to enable multiprocessing
        self.dataset_name: str = dataset_name

        self.data_set_dynamic = np.load("Datapoints.npy", allow_pickle=True)  # loading datapoints from the "Datapoints.npy" file.

        self.lower_threshold_colour: float = 0.0  # colour lower limit (for linear transform from 0.0 to ___)
        self.higher_threshold_colour: float = 0.5  # colour upper limit (for linear transform from 1.0 to ___)

    def warning_bounds(self) -> int:
        """
        Check if the bounds_buffer covers the grid edges to avoid data loss.

        Returns:
            int: -1 if bounds_buffer < grid_space * grid_divide (risk of loss),
                 0 otherwise.
        """

        if self.bounds_buffer < self.grid_space * self.grid_divide:  # if the buffer value is less than the grid spacing times the grid divisions then there could be data loss at the end part of the axes.

            print("WARNING: BUFFER VALUE TOO LOW, MAY LEAD TO LOSSES.")

            return -1

        return 0

    def calculate_grid_information(self) -> None:
        """
        Calculate grid information and write two .npy files:
        
        Outputs:
            Grid_Info.npy   (1×4 float array): [spacing, buffer, divisions, dimension]
            Grid_Bounds.npy (dim×2 float array): lower/upper bound per axis
        
        Returns:
            None
        """

        grid_dimension = len(self.data_set_dynamic[0][0]) - 1  # Getting the grid dimension of the data present based on a single vector in data_set_dynamic.
        # Last parameter of the vector is label/identity of the point/vector, so excluding that from the dimension.

        warn_value = self.warning_bounds()  # checking if bounds buffer is sufficiently high.

        if warn_value == -1:  # in case of low bounds warning

            self.bounds_buffer = self.grid_space * self.grid_divide + sys.float_info.min  # assigning minimum value to buffer.

            print("AUTOMATIC BOUND BUFFER CHANGE TO PREVENT LOSSES")

        grid_info = [self.grid_space, self.bounds_buffer, self.grid_divide, grid_dimension]  # grid information list

        np.save("Grid_Info.npy", np.asarray(grid_info))  # saving to file Grid_Info.npy

        datapoints = np.vstack(self.data_set_dynamic)  # stacking the structures together to simply get all points in a big 2D array for computation.

        # grid bounds calculation of each dimension
        grid_bounds = [np.min(datapoints[:, : grid_dimension], axis=0) - self.bounds_buffer, np.max(datapoints[:, : grid_dimension], axis=0) + self.bounds_buffer]
        # transposing the above 2D array to get the bounds in the desired format.
        grid_bounds = np.asarray(grid_bounds).transpose()

        np.save("Grid_Bounds", grid_bounds)  # saving to file Grid_Bounds.npy

        return None

    def compute_grid(self) -> None:
        """
        Compute the grid structure using the assigned algorithm.

        Prerequisite:
            calculate_grid_information() must be called first to generate Grid_Info.npy and Grid_Bounds.npy.

        Side-effects:
            Instantiates GridGenerator, sets algorithm, and computes/saves grid metadata.

        Returns:
            None
        """

        gg = GridGenerator(self.algorithm)
        gg.algorithm_set()
        gg.compute()

        return None

    @staticmethod
    def compute_boundary_neighbour(weight: float = 0.5, neighbour_set_method: bool = False) -> None:
        """
        Compute object boundaries using the NCubeNeighbours algorithm.

        Prerequisite:
            compute_grid() must be called first to generate required grid metadata.

        Parameters:
            weight (float): weight factor for lower vs higher structure identity (default=0.5).
            neighbour_set_method (bool): select neighbor set method (default=False).

        Returns:
            None
        """

        BoundaryExtractor.extract_neighbour(weight=weight, neighbours_set_method=neighbour_set_method)

        return None

    @staticmethod
    def fractal_detection() -> None:
        """
        Perform fractal detection using box‑counting algorithm.

        Prerequisite:
            compute_boundary_neighbour() must be called first to generate boundaries.

        Returns:
            None
        """

        fd = FractalDetector()
        fd.detect()

        return None

    def display(self) -> None:
        """
            Generate boundary overlay plots for data and grid.

            Prerequisite:
                compute_boundary_neighbour() must be called first to produce Neighbour_Boundary.npy.

            Outputs:
            - Data_Points_Plot_Boundary.png: raw points overlaid with neighbor boundaries
            - Grid_Plot_Boundary.png: grid points overlaid with neighbor boundaries
            - Boundary_Plot.png: neighbor boundaries only

        Returns:
            None

        Note:
            If boundary output seems stale, delete any existing Boundary_Type.npy before rerunning.
        """

        try:
            boundary_type = np.load("Boundary_Type.npy")  # checking for existence
        except FileNotFoundError:
            FractalDetector().detect(create_boundary_type_only=True)  # creation of only the required file.
            boundary_type = np.load("Boundary_Type.npy")

        boundary_type_labels = []
        boundaries = []
        colour_bound = []

        for i in range(len(boundary_type)):

            bound = BoundaryExtractor.fetch(mini_grid_num=-1, lower_struct=boundary_type[i, 0], higher_struct=boundary_type[i, 1], filter_boundary=True)
            if len(bound) != 0:
                boundaries.append(bound)
                colour_bound.append(self.lower_threshold_colour + (self.higher_threshold_colour - self.lower_threshold_colour) * np.random.rand(3))
                boundary_type_labels.append(boundary_type[i])

        grid_points = np.load("Grid_Points.npy")

        fig, ax = plt.subplots()  # getting the Figure and Axes objects
        fig.suptitle(f"({self.dataset_name}) Data Points Plot with Boundary")  # Setting the title of the Figure
        ax.set_xlabel("X Coordinates")  # Set the X axis label
        ax.set_ylabel("Y Coordinates")  # Set the Y axis label

        i: int = 0  # loop control variable

        while i < len(self.data_set_dynamic):  # parsing through the structures

            color = self.lower_threshold_colour + (self.higher_threshold_colour - self.lower_threshold_colour) * np.random.rand(3)  # generating a random color to represent datapoints of that structure
            struct_data = self.data_set_dynamic[i]  # extracting i th structure's data.
            ax.scatter(struct_data[:, 0], struct_data[:, 1], label=f"Class {i + 1}",c=[color] * len(struct_data),
                       marker=".", s=0.01)  # plotting points of the circle with a color and dot marker.
            i += 1  # update statement

        i: int = 0  # loop control variable

        while i < len(boundaries):  # parsing through the structures

            bound = boundaries[i]  # extracting i th structure's data.
            bound = np.vstack(bound)

            ax.scatter(bound[:, 0], bound[:, 1], label=f"Boundary {boundary_type_labels[i][0]}-{boundary_type_labels[i][1]}",c=[colour_bound[i]] * len(bound), marker=".", s=0.01)
            i += 1  # update statement

        # Get the handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()

        # Adjust the legend to show larger marker sizes
        ax.legend(handles, labels, scatterpoints=1, markerscale=100, handletextpad=0.5, prop={'size': 8}, loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
        plt.subplots_adjust(right=0.8)
        plt.savefig("Data_Points_Plot_Boundary.png")  # Saving the System as the image "Data_Points_Plot.png"
        # plt.show()  # show the system on console
        plt.close()

        # ..........

        fig, ax = plt.subplots()  # getting the Figure and Axes objects
        fig.suptitle(f"({self.dataset_name}) Boundary Plot")  # Setting the title of the Figure
        ax.set_xlabel("X Coordinates")  # Set the X axis label
        ax.set_ylabel("Y Coordinates")  # Set the Y axis label

        i: int = 0  # loop control variable

        while i < len(boundaries):  # parsing through the mini grid divided boundary

            bound = boundaries[i]  # extracting i th available grid.
            bound = np.vstack(bound)

            ax.scatter(bound[:, 0], bound[:, 1], label=f"Boundary {boundary_type_labels[i][0]}-{boundary_type_labels[i][1]}", c=[colour_bound[i]] * len(bound), marker=".", s=0.01)
            i += 1  # update statement

        # Get the handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()

        # Adjust the legend to show larger marker sizes
        ax.legend(handles, labels, scatterpoints=1, markerscale=100, handletextpad=0.5, prop={'size': 8}, loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
        plt.subplots_adjust(right=0.8)
        plt.savefig("Boundary_Plot.png")  # Saving the System as the image "Data_Points_Plot.png"
        # plt.show()  # show the system on console
        plt.close()

        # ..........

        fig, ax = plt.subplots()
        fig.suptitle(f"({self.dataset_name}) Grid Plot with Boundary")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        colour = []

        for i in range(len(self.data_set_dynamic)):
            colour.append(self.lower_threshold_colour + (self.higher_threshold_colour - self.lower_threshold_colour) * np.random.rand(3))

        for grid in grid_points:

            i: int = 0

            while i < len(self.data_set_dynamic):

                grid_data_i = grid[
                    grid[:, 2] == i + 1]  # extracting datapoints having the same label in the grid via slicing

                ax.scatter(grid_data_i[:, 0], grid_data_i[:, 1], label=f"Grid Class {i + 1}", c=[colour[i]] * len(grid_data_i), marker=".", s=0.01)
                i += 1

        i: int = 0  # loop control variable

        while i < len(boundaries):  # parsing through the structures

            bound = boundaries[i]  # extracting i th structure's data.
            bound = np.vstack(bound)

            ax.scatter(bound[:, 0], bound[:, 1], label=f"Boundary {boundary_type_labels[i][0]}-{boundary_type_labels[i][1]}", c=[colour_bound[i]] * len(bound), marker=".", s=0.01)
            i += 1  # update statement

        # Create custom legend for grid points (larger markers) and boundaries (lines)
        legend_handles = [
            Line2D([0], [0], marker='.', color=tuple(colour[i]), markersize=10, linestyle='None',
                   label=f"Grid Class {i + 1}")
            for i in range(len(colour))
        ]

        for i in range(len(boundary_type_labels)):

            legend_handles.append(Line2D([0], [0], marker='.', color=tuple(colour_bound[i]), markersize=10, linestyle='None', label=f"Boundary {boundary_type_labels[i][0]}-{boundary_type_labels[i][1]}"))

        # Add the legend with unique labels
        ax.legend(handles=legend_handles, scatterpoints=1, markerscale=1, handletextpad=0.5, prop={'size': 8}, loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
        plt.subplots_adjust(right=0.8)
        plt.savefig("Grid_Plot_Boundary.png")
        # plt.show()
        plt.close()

        return None

    def display_data(self) -> None:
        """
        Generate a scatter plot of raw data points.

        Prerequisite:
            Ensure `self.data_set_dynamic` is populated.

        Outputs:
            Data_Points_Plot.png: scatter plot saved of all raw data points.

        Returns:
            None
        """

        fig, ax = plt.subplots()  # getting the Figure and Axes objects
        fig.suptitle(f"({self.dataset_name}) Data Points Plot")  # Setting the title of the Figure
        ax.set_xlabel("X Coordinates")  # Set the X axis label
        ax.set_ylabel("Y Coordinates")  # Set the Y axis label

        i: int = 0  # loop control variable

        while i < len(self.data_set_dynamic):  # parsing through the structures

            color = self.lower_threshold_colour + (self.higher_threshold_colour - self.lower_threshold_colour) * np.random.rand(3)  # generating a random color to represent datapoints of that structure
            struct_data = self.data_set_dynamic[i]  # extracting i th structure's data.
            ax.scatter(struct_data[:, 0], struct_data[:, 1], label=f"Class {i + 1}", c=[color] * len(struct_data),
                       marker=".", s=0.01)  # plotting points of the circle with a color and dot marker.
            i += 1  # update statement

        # Get the handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()

        # Adjust the legend to show larger marker sizes
        ax.legend(handles, labels, scatterpoints=1, markerscale=100, handletextpad=0.5, prop={'size': 8}, loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
        plt.subplots_adjust(right=0.8)
        plt.savefig("Data_Points_Plot.png")  # Saving the System as the image "Data_Points_Plot.png"
        # plt.show()  # show the system on console
        plt.close()

        return None

    def display_grid(self) -> None:
        """
        Generate grid visualizations overlaying raw data.

        Prerequisite:
            calculate_grid_information() and compute_grid() must be called first.

        Outputs:
            - Grid_Plot_Datapoints.png: overlay of raw points and grid points.
            - Grid_Plot.png: plot of grid points only.

        Returns:
            None

        Note:
            Ensure Grid_Points.npy exists before calling this method.
        """

        grid_points = np.load("Grid_Points.npy")
        colour_data = []

        fig, ax = plt.subplots()
        fig.suptitle(f"({self.dataset_name}) Grid Plot")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        colour = []

        for i in range(len(self.data_set_dynamic)):
            colour.append(self.lower_threshold_colour + (self.higher_threshold_colour - self.lower_threshold_colour) * np.random.rand(3))

        for grid in grid_points:

            i: int = 0

            while i < len(self.data_set_dynamic):
                grid_data_i = grid[
                    grid[:, 2] == i + 1]  # extracting datapoints having the same label in the grid via slicing

                ax.scatter(grid_data_i[:, 0], grid_data_i[:, 1], label=f"Grid Class {i + 1}", c=[colour[i]] * len(grid_data_i), marker=".", s=0.01)
                i += 1

        legend_handles = [
            Line2D([0], [0], marker='.', color=tuple(colour[i]), markersize=10, linestyle='None',
                   label=f"Grid Class {i + 1}")
            for i in range(len(colour))
        ]

        # Add the legend with unique labels
        ax.legend(handles=legend_handles, scatterpoints=1, markerscale=1, handletextpad=0.5, prop={'size': 8}, loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
        plt.subplots_adjust(right=0.8)
        plt.savefig("Grid_Plot.png")
        # plt.show()
        plt.close()

        # ..........

        fig, ax = plt.subplots()
        fig.suptitle(f"({self.dataset_name}) Grid Plot with Datapoints")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        colour = []

        for i in range(len(self.data_set_dynamic)):
            colour.append(self.lower_threshold_colour + (self.higher_threshold_colour - self.lower_threshold_colour) * np.random.rand(3))

        for grid in grid_points:

            i: int = 0

            while i < len(self.data_set_dynamic):
                grid_data_i = grid[
                    grid[:, 2] == i + 1]  # extracting datapoints having the same label in the grid via slicing

                ax.scatter(grid_data_i[:, 0], grid_data_i[:, 1], label=f"Grid Class {i + 1}", c=[colour[i]] * len(grid_data_i), marker=".", s=0.01)
                i += 1

        i: int = 0  # loop control variable

        while i < len(self.data_set_dynamic):  # parsing through the structures

            color = self.lower_threshold_colour + (self.higher_threshold_colour - self.lower_threshold_colour) * np.random.rand(3)  # generating a random color to represent datapoints of that structure
            colour_data.append(color)
            struct_data = self.data_set_dynamic[i]  # extracting i th structure's data.
            ax.scatter(struct_data[:, 0], struct_data[:, 1], label=f"Class {i + 1}", c=[color] * len(struct_data),
                       marker=".", s=0.01)  # plotting points of the circle with a color and dot marker.
            i += 1  # update statement

        # Create custom legend for grid points (larger markers) and boundaries (lines)
        legend_handles = [
            Line2D([0], [0], marker='.', color=tuple(colour[i]), markersize=10, linestyle='None', label=f"Grid Class {i + 1}")
            for i in range(len(colour))
        ]

        for i in range(len(self.data_set_dynamic)):

            legend_handles.append(Line2D([0], [0], marker='.', color=tuple(colour_data[i]), markersize=10, linestyle='None', label=f"Class {i + 1}"))

        # Add the legend with unique labels
        ax.legend(handles=legend_handles, scatterpoints=1, markerscale=1, handletextpad=0.5, prop={'size': 8}, loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
        plt.subplots_adjust(right=0.8)
        plt.savefig("Grid_Plot_Datapoints.png")
        # plt.show()
        plt.close()

        return None

    def display_object_grid(self, identity: int, direct_computation_ctrl: bool = False, perc_range_lower: float = -1, perc_range_upper: float = -1) -> None:
        """
        Generate object-specific grid overlay plots.

        Prerequisite:
            `ForceGridObject.compute()` must be called first to generate `Grid_Points_Object_{identity}.npy`.

        Parameters:
            identity (int): identifier of the target object.
            direct_computation_ctrl (bool): if True, use direct computation of object grid points.
            perc_range_lower (float): lower percentage threshold for object filtering.
            perc_range_upper (float): upper percentage threshold for object filtering.

        Outputs:
            - Grid_Plot_Datapoints_Object_{id}.png: overlay of raw and object grid points.
            - Grid_Plot_Object_{id}.png: plot of object grid points only.

        Returns:
            None
        """

        if perc_range_lower == -1 or perc_range_upper == -1:
            grid_points = np.load(f"Grid_Points_Object_{identity}.npy")
        else:
            grid_points = np.load(f"Grid_Points_Object_{identity}_{perc_range_lower}-{perc_range_upper}.npy")

        fig, ax = plt.subplots()

        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        colour = np.asarray([1.0, 0.0, 0.0])  # Displaying in Red

        count = 0

        if not direct_computation_ctrl:

            for grid in grid_points:

                grid_data_i = grid[
                    grid[:, 2] == identity]  # extracting datapoints having the same label in the grid via slicing

                count += len(grid_data_i)
                ax.scatter(grid_data_i[:, 0], grid_data_i[:, 1], label=f"Grid Class {identity}",
                           c=[colour] * len(grid_data_i), marker=".", s=1)

        else:

            ax.scatter(grid_points[:, 0], grid_points[:, 1], label=f"Grid Class {identity}",
                       c=[colour] * len(grid_points), marker=".", s=1)

            count = len(grid_points)

        fig.suptitle(f"({self.dataset_name}) Grid Plot Object {identity} - {count}")
        legend_handles = [
            Line2D([0], [0], marker='.', color=tuple(colour), markersize=10, linestyle='None',
                   label=f"Grid Class {identity}")
        ]

        # Add the legend with unique labels
        ax.legend(handles=legend_handles, scatterpoints=1, markerscale=1, handletextpad=0.5, prop={'size': 8},
                  loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
        plt.subplots_adjust(right=0.8)
        plt.savefig(f"Grid_Plot_Object_{identity}.png")
        # plt.show()
        plt.close()

        # ..........

        fig, ax = plt.subplots()
        fig.suptitle(f"({self.dataset_name}) Grid Plot with Datapoints Object {identity}")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        colour = np.asarray([1.0, 0.0, 0.0])  # Using Red Colour

        if not direct_computation_ctrl:

            for grid in grid_points:

                grid_data_i = grid[
                    grid[:, 2] == identity]  # extracting datapoints having the same label in the grid via slicing

                count += len(grid_data_i)
                ax.scatter(grid_data_i[:, 0], grid_data_i[:, 1], label=f"Grid Class {identity}",
                           c=[colour] * len(grid_data_i), marker=".", s=1)

        else:

            ax.scatter(grid_points[:, 0], grid_points[:, 1], label=f"Grid Class {identity}",
                       c=[colour] * len(grid_points), marker=".", s=1)

        struct_data = self.data_set_dynamic[identity - 1]  # extracting i th structure's data.
        ax.scatter(struct_data[:, 0], struct_data[:, 1], label=f"Class {identity}", c='blue',
                   marker=".", s=1, alpha=0.25)  # plotting points of the circle with a color and dot marker.

        # Create custom legend for grid points (larger markers) and boundaries (lines)
        legend_handles = [Line2D([0], [0], marker='.', color=tuple(colour), markersize=10, linestyle='None',
                                 label=f"Grid Class {identity}"),
                          Line2D([0], [0], marker='.', color='blue', markersize=10, linestyle='None',
                                 label=f"Class {identity}", alpha=0.25)]

        # Add the legend with unique labels
        ax.legend(handles=legend_handles, scatterpoints=1, markerscale=1, handletextpad=0.5, prop={'size': 8},
                  loc='upper left', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
        plt.subplots_adjust(right=0.8)
        plt.savefig(f"Grid_Plot_Datapoints_Object_{identity}.png")
        # plt.show()
        plt.close()

        return None

    @staticmethod
    def display_hatch(identity: int) -> None:
        """
        Generate a hatch‑point scatter plot for a specific object.

        Prerequisite:
            ForceGridObject.compute() must be called first to save `Hatch_Object_{identity}.npy`.

        Parameters:
            identity (int): identifier of the object whose hatch points to plot.

        Outputs:
            Hatch_Object_{identity}.png: scatter plot of hatch points from `Hatch_Object_{identity}.npy`.

        Returns:
            None
        """

        grid_points = np.load(f"Hatch_Object_{identity}.npy")

        fig, ax = plt.subplots()
        fig.suptitle(f"Hatch_Object_{identity}")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        for grid in grid_points:

            ax.scatter(grid[:, 0], grid[:, 1],
                       c='black', marker=".", s=0.01)

        plt.savefig(f"Hatch_Object_{identity}.png")
        # plt.show()
        plt.close()

        return None

    @staticmethod
    def display_time_analysis() -> None:
        """
        Plot timing metrics for grid, boundary, fractal, connectivity, and hatch analyses.

        Prerequisite:
            All `Time_*.npy` arrays and `X_Grid_Space.npy`/`X_Grid_Divide.npy` must be generated 
            (e.g., via TimeAnalysis.compute() in TimeAnalysis.py).

        Outputs:
            - Time_Grid_Spacing.png & Time_Grid_Divisions.png
            - Time_Boundary_Spacing.png & Time_Boundary_Divisions.png
            - Time_Fractal_Spacing.png & Time_Fractal_Divisions.png
            - Time_Connectivity_Spacing.png & Time_Connectivity_Divisions.png
            - Time_Force_Grid_Spacing.png & Time_Force_Grid_Divisions.png
            - Time_Hatch_Connectivity_Spacing.png & Time_Hatch_Connectivity_Divisions.png
            - Time_Force_Grid_Connectivity_Spacing.png & Time_Force_Grid_Connectivity_Divisions.png

        Returns:
            None

        Note:
            The TimeAnalysis.py module is not up to date—verify or update its methods 
            before regenerating these time data files.
        """

        time_grid = np.load("Time_Grid.npy", allow_pickle=True)
        time_boundary = np.load("Time_Boundary.npy", allow_pickle=True)
        time_fractal = np.load("Time_Fractal.npy", allow_pickle=True)
        time_connectivity = np.load("Time_Connectivity.npy", allow_pickle=True)
        time_force_grid = np.load("Time_Force_Grid.npy", allow_pickle=True)
        time_hatch_connectivity = np.load("Time_Hatch_Connectivity.npy", allow_pickle=True)
        time_force_grid_connectivity = np.load("Time_Force_Grid_Connectivity.npy", allow_pickle=True)

        x_grid_divide = np.load("X_Grid_Divide.npy")
        x_grid_space = np.load("X_Grid_Space.npy")

        Manager.plot_x_y_time(x_grid_space, time_grid[0], "Spacing", "Grid")
        Manager.plot_x_y_time(x_grid_divide, time_grid[1], "Divisions", "Grid")

        Manager.plot_x_y_time(x_grid_space, time_boundary[0], "Spacing", "Boundary")
        Manager.plot_x_y_time(x_grid_divide, time_boundary[1], "Divisions", "Boundary")

        Manager.plot_x_y_time(x_grid_space, time_fractal[0], "Spacing", "Fractal")
        Manager.plot_x_y_time(x_grid_divide, time_fractal[1], "Divisions", "Fractal")

        Manager.plot_x_y_time(x_grid_space, time_connectivity[0], "Spacing", "Connectivity")
        Manager.plot_x_y_time(x_grid_divide, time_connectivity[1], "Divisions", "Connectivity")

        Manager.plot_x_y_time(x_grid_space, time_force_grid[0], "Spacing", "Force_Grid")
        Manager.plot_x_y_time(x_grid_divide, time_force_grid[1], "Divisions", "Force_Grid")

        Manager.plot_x_y_time(x_grid_space, time_hatch_connectivity[0], "Spacing", "Hatch_Connectivity")
        Manager.plot_x_y_time(x_grid_divide, time_hatch_connectivity[1], "Divisions", "Hatch_Connectivity")

        Manager.plot_x_y_time(x_grid_space, time_force_grid_connectivity[0], "Spacing", "Force_Grid_Connectivity")
        Manager.plot_x_y_time(x_grid_divide, time_force_grid_connectivity[1], "Divisions", "Force_Grid_Connectivity")

        return None

    @staticmethod
    def plot_x_y_time(x: np.ndarray, y: np.ndarray, name_x: str, type_graph: str) -> None:
        """
        Plot a metric over time and save the figure.

        Parameters:
            x (np.ndarray): X‑axis values (e.g., grid spacing or divisions).
            y (np.ndarray): Y‑axis values (timing data).
            name_x (str): label for X‑axis in filename and title.
        type_graph (str): metric name for filename and title.

        Outputs:
            Time_{type_graph}_{name_x}.png: line-and-scatter plot saved to disk.

        Returns:
            None
        """

        fig, ax = plt.subplots()
        fig.suptitle(f"Time vs {name_x} ({type_graph}) Graph")
        ax.set_xlabel(f"{name_x} Coordinates")
        ax.set_ylabel("Time Coordinates")

        ax.scatter(x, y, c='black', marker="o", s=1)

        ax.plot(x, y, color='red', linewidth=1)

        plt.savefig(f"Time_{type_graph}_{name_x}.png")
        # plt.show()
        plt.close()

        return None

    @staticmethod
    def connectivity_deterioration_display(identity: int) -> None:
        """
        Plot object deterioration metrics over iterations.

        Prerequisite:
            ConnectivityDeterioration.connectivity_deterioration_analysis() must save `Timings_Object_{identity}.npy`, 
            `Lengths_Object_{identity}.npy`, and `CFS_Object_{identity}.npy`.

        Parameters:
            identity (int): identifier of the target object.

        Outputs:
            - Time_Length_Object_{identity}.png: time vs. object length.
            - Time_Connectivity_Object_{identity}.png: time vs. connectivity factor.
            - Length_Connectivity_Object_{identity}.png: length vs. connectivity factor.
            - Iterations_Connectivity_Object_{identity}.png: iterations vs. connectivity factor.

        Returns:
            None

        Note:
            Lengths are log‑scaled for readability. Timing data may be unevenly spaced.
        """

        time_points = np.load(f"Timings_Object_{identity}.npy")
        length_points = np.load(f"Lengths_Object_{identity}.npy")
        cfs_points = np.load(f"CFS_Object_{identity}.npy")
        iterations = np.arange(0, len(time_points), 1)

        fig, ax = plt.subplots()
        fig.suptitle(f"Log(Length) vs Time Object {identity}")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        ax.scatter(time_points, np.log(length_points),c='black', marker=".", s=1)
        ax.plot(time_points, np.log(length_points), color='red', linewidth=1)

        plt.savefig(f"Time_Length_Object_{identity}.png")
        # plt.show()
        plt.close()

        fig, ax = plt.subplots()
        fig.suptitle(f"Connectivity vs Time Object {identity}")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        ax.scatter(time_points, cfs_points, c='black', marker=".", s=1)
        ax.plot(time_points, cfs_points, color='red', linewidth=1)

        plt.savefig(f"Time_Connectivity_Object_{identity}.png")
        # plt.show()
        plt.close()

        fig, ax = plt.subplots()
        fig.suptitle(f"Connectivity vs Log(Length) Object {identity}")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        ax.scatter(np.log(length_points), cfs_points, c='black', marker=".", s=1)
        ax.plot(np.log(length_points), cfs_points, color='red', linewidth=1)

        plt.savefig(f"Length_Connectivity_Object_{identity}.png")
        # plt.show()
        plt.close()

        fig, ax = plt.subplots()
        fig.suptitle(f"Connectivity vs Iterations Object {identity}")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")

        ax.scatter(iterations, cfs_points, c='black', marker=".", s=1)
        ax.plot(iterations, cfs_points, color='red', linewidth=1)

        plt.savefig(f"Iterations_Connectivity_Object_{identity}.png")
        # plt.show()
        plt.close()

        return None

    @staticmethod
    def analysis_object_display(index: int, force_identity: list) -> None:
        """
        Plot connectivity versus percentage and log-spacing for a specific object analysis of connectivity factor.

        Parameters:
            index (int): index of the object in the `force_identity` list.
            force_identity (list[int]): list of object identifiers.

        Outputs:
            CF_Percentage_Object_{id}.png: connectivity vs percentage plot.
            CF_Spacing_Object_{id}.png: connectivity vs ln(spacing) plot.

        Returns:
            None
        """
        
        fig, ax = plt.subplots()
        fig.suptitle(f"Connectivity vs Percentage Object {force_identity[index]}")
        ax.set_xlabel("Percentage Coordinates")
        ax.set_ylabel("CF Coordinates")

        analysis = np.load("Analysis.npy", allow_pickle=True)
        analysis = np.array(analysis[index])
        analysis = analysis.astype(np.float64)  # Convert to proper numeric type

        ax.plot(analysis[:, 4], analysis[:, 3], color='red', linewidth=1)
        ax.scatter(analysis[:, 4], analysis[:, 3], c='black', marker=".", s=15)

        plt.savefig(f"CF_Percentage_Object_{force_identity[index]}.png")
        plt.close()

        fig, ax = plt.subplots()
        fig.suptitle(f"Connectivity vs ln(Spacing) Object {force_identity[index]}")
        ax.set_xlabel("ln(Spacing) Coordinates")
        ax.set_ylabel("CF Coordinates")

        ax.plot(np.log(analysis[:, 2]), analysis[:, 3], color='red', linewidth=1)
        ax.scatter(np.log(analysis[:, 2]), analysis[:, 3], c='black', marker=".", s=15)

        plt.savefig(f"CF_Spacing_Object_{force_identity[index]}.png")
        plt.close()

        return None

    @staticmethod
    def weights_object_display(index: int, force_identity: list) -> None:
        """
        Plot normalized weight distributions against data percentage for a specific object.

        Parameters:
            index (int): index of the target object in the `force_identity` list.
            force_identity (list[int]): list of object identifiers.

        Outputs:
            Weights_Percentage_Object_{id}.png: normalized weights vs. percentage scatter/line plot.

        Returns:
            None
        """

        weights_analysis = np.load("Weights_Analysis.npy", allow_pickle=True)
        weights_analysis = np.array(weights_analysis[index])

        analysis = np.load("Analysis.npy", allow_pickle=True)
        analysis = np.array(analysis[index])
        analysis = analysis.astype(np.float64)  # Convert to proper numeric type

        x = analysis[:, 4]

        weights_perc_norm = []

        for weights in weights_analysis:

            weights_perc_arr = []

            weights_sum = 0.0

            weights_iter = iter(weights.items())

            for key, value in weights_iter:
                weights_sum += value

                weights_perc_arr.append(value)

            weights_perc_arr = np.array(weights_perc_arr)

            weights_perc_arr = weights_perc_arr / weights_sum

            weights_perc_norm.append(weights_perc_arr)

        weights_perc_norm = np.array(weights_perc_norm)

        Manager.plot_weights_vs_percentage(weights_perc_norm, x, object_id=f'{force_identity[index]}')

        return None

    @staticmethod
    def plot_weights_vs_percentage(
        weights_perc_norm: np.ndarray,
        x: np.ndarray,
        title: str = "Weights vs Percentage",
        xlabel: str = "Percentage Coordinates",
        ylabel: str = "Normalized Weights",
        object_id: str = None # Optional identifier for the title
    ) -> None:
        """
        Plot normalized weight distributions against data percentage and save the figure.

        Parameters:
            weights_perc_norm (np.ndarray): 2D array (n_points, n_weight_sets) of normalized weights.
            x (np.ndarray): 1D array of percentage values, length n_points.
            title (str): base title for the plot.
            xlabel (str): label for the x-axis.
            ylabel (str): label for the y-axis.
            object_id (str, optional): identifier to include in the plot title and filename.

        Outputs:
            Weights_Percentage_Object_{object_id}.png: saved scatter-and-line plot.

        Returns:
            None
        """
        
        # --- Input Validation ---
        if not isinstance(weights_perc_norm, np.ndarray) or weights_perc_norm.ndim != 2:
            print(f"Error: weights_perc_norm must be a 2D NumPy array. Got shape {getattr(weights_perc_norm, 'shape', 'N/A')}")
            return
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            print(f"Error: x must be a 1D NumPy array. Got shape {getattr(x, 'shape', 'N/A')}")
            return
        if weights_perc_norm.shape[0] != x.shape[0]:
            print(f"Error: Number of rows in weights_perc_norm ({weights_perc_norm.shape[0]}) "
                  f"must match the number of elements in x ({x.shape[0]}).")
            return
        if weights_perc_norm.shape[1] == 0:
            print("Error: weights_perc_norm has no columns to plot.")
            return

        # --- Plotting Setup ---
        fig, ax = plt.subplots(figsize=(12, 8))

        # Construct title
        plot_title = title
        if object_id is not None:
            plot_title += f" (Object: {object_id})"

        fig.suptitle(plot_title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        num_lines = weights_perc_norm.shape[1]

        # --- Plotting Loop ---
        for i in range(num_lines):
            y_column = weights_perc_norm[:, i] # Extract the i-th column

            # Handle potential NaNs if they exist from previous processing
            # Plot only non-NaN values to avoid gaps or errors
            mask = ~np.isnan(y_column)
            if not np.any(mask): # Skip column if all values are NaN
                print(f"Warning: Weight Set {i+1} contains only NaN values. Skipping.")
                continue

            x_valid = x[mask]
            y_valid = y_column[mask]

            if len(x_valid) == 0: # Double check after masking
                 print(f"Warning: Weight Set {i+1} has no valid (non-NaN) data points after filtering. Skipping.")
                 continue

            # Plot the line
            line, = ax.plot(x_valid, y_valid, label=f'm={i}', alpha=0.75, linewidth=1.5)
            line_color = line.get_color()

            # Calculate darker color
            try:
                rgb = mcolors.to_rgb(line_color)
                h, l, s = colorsys.rgb_to_hls(*rgb)
                darker_l = max(0., l * 0.6) # Adjust 0.6 factor as needed
                darker_rgb = colorsys.hls_to_rgb(h, darker_l, s)
                darker_rgb = tuple(np.clip(c, 0, 1) for c in darker_rgb) # Clip for safety
            except ValueError:
                print(f"Warning: Could not darken color {line_color} for set {i+1}. Using line color for markers.")
                darker_rgb = line_color

            # Plot the markers
            ax.scatter(x_valid, y_valid, color=darker_rgb, s=30, zorder=line.get_zorder() + 1)

        # --- Final Touches ---
        if ax.has_data(): # Only add legend/grid if something was plotted
            ax.legend(title="Weight Sets")
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, "No data to display", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plt.savefig(f"Weights_Percentage_Object_{object_id}.png")
        plt.close()

        return None