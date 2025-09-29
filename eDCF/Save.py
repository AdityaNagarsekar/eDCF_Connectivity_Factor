import os, shutil
from typing import List


class Save:
    """
    Manage saving analysis outputs to organized directories.

    Provides methods to save plots, NumPy arrays, and reports,
    with optional clearing of source files.

    Attributes
    ----------
    folder_name : str
        Target directory for saving files.
    dataset_name : str
        Prefix for saved filenames.

    Methods
    -------
    save(clear=False) -> None
        Save general analysis files.
    save_force(identity, clear=False) -> None
        Save force grid object outputs for a specific identity.
    save_force_connectivity(clear=False) -> None
        Save connectivity-related force grid outputs.
    save_time_analysis(clear=False) -> None
        Save timing analysis plots and data.
    """

    def __init__(self, dataset_name: str, directory_name: str="Save File"):
        """
        Initialize Save utility.

        Parameters
        ----------
        dataset_name : str
            Prefix for filenames, e.g., '(dataset_name)_<file>'.
        directory_name : str, optional
            Directory where files will be stored (default: 'Save File').
        """

        self.folder_name = directory_name
        self.dataset_name = dataset_name

    def save(self, clear: bool = False) -> None:
        """
        Save general output files to the target folder.

        Parameters
        ----------
        clear : bool, optional
            If True, remove source files instead of moving them (default: False).

        Returns
        -------
        None

        Notes
        -----
        Attempts to save predefined plots, NumPy arrays, and reports.
        Missing files are skipped.
        """

        os.makedirs(self.folder_name, exist_ok=True)

        source_files: List[str] = [ "Data_Points_Plot.png",
                                   "Grid_Plot.png",
                                   "Grid_Plot_Datapoints.png",
                                   "Boundary_Plot.png",
                                   "Data_Points_Plot_Boundary.png",
                                   "Grid_Plot_Boundary.png",
                                   "Datapoints.npy",
                                   "Grid_Info.npy",
                                   "Grid_Bounds.npy",
                                   "Divided_Grid_Bounds.npy",
                                   "Grid_Points.npy",
                                   "Neighbour_Boundary.npy",
                                   "Boundary_Type.npy",
                                   "Dimensions.npy",
                                    "Connectivity_Factors.npy",
                                   "Report.txt",
                                    "Topological_Dimensions_Boundaries.npy",
                                    "Topological_Dimensions_Object.npy",
                                    "Topological_Dimensions_Hatch.npy",
                                    "Analysis.npy",
                                    "Fractal_Dimensions_Objects.npy",
                                    "Weights_Analysis.npy",
                                    "Weights_Boundaries.npy",
                                    "Weights_Force.npy",
                                    "Weighted_Topological_Dimensions_Boundaries.npy",
                                    "Weights_Boundaries_Normalized.npy",
                                    "Weighted_Topological_Dimensions_Hatch.npy",
                                    "Weights_Hatch_Normalized.npy",
                                    "Weighted_Topological_Dimensions_Force.npy",
                                    "Weights_Force_Normalized.npy",
                                    "Weighted_Topological_Dimensions_Force.npy",
                                    "Weights_Force_Normalized.npy"]

        for i, source_file in enumerate(source_files):

            final_name = f"({self.dataset_name})_{source_file}"
            try:
                os.rename(source_file, final_name)
            except FileNotFoundError:
                pass
            source_files[i] = final_name

        for i in range(len(source_files)):
            if clear:
                try:
                    os.remove(source_files[i])
                except FileNotFoundError:
                    pass

            try:
                if source_files[i] == f"({self.dataset_name})_Datapoints.npy":
                    shutil.copy(source_files[i], os.path.join(self.folder_name, os.path.basename(source_files[i])))
                    os.rename(source_files[i], "Datapoints.npy")
                else:
                    shutil.move(source_files[i], os.path.join(self.folder_name, f"{source_files[i]}"))
            except FileNotFoundError:
                pass

        source_files_algorithms: List[str] = ["Hyper_Param.npy",
                                              "Testing_Score.npy",
                                              "X_train.npy",
                                              "X_test.npy",
                                              "y_train.npy",
                                              "y_test.npy"]

        for i, source_file in enumerate(source_files_algorithms):

            final_name = f"algorithms/({self.dataset_name})_{source_file}"
            try:
                os.rename(source_file, final_name)
            except FileNotFoundError:
                pass
            source_files_algorithms[i] = final_name

        for i in range(len(source_files_algorithms)):

            if clear:
                try:
                    os.remove(source_files[i])
                except FileNotFoundError:
                    pass

            file_name: str = source_files_algorithms[i][len("algorithms/"):]
            try:
                shutil.move(source_files_algorithms[i], os.path.join(self.folder_name, f"{file_name}"))
            except FileNotFoundError:
                pass

    def save_force(self, identity: int, clear: bool = False) -> None:
        """
        Save force grid object outputs for a given identity.

        Parameters
        ----------
        identity : int
            Identifier of the force grid object.
        clear : bool, optional
            If True, remove source files instead of moving them (default: False).

        Returns
        -------
        None

        Notes
        -----
        Saves plots and NumPy arrays related to force grid analysis.
        """

        sub_folder_name = self.folder_name + f"/Force_Grid_Object_{identity}"

        os.makedirs(sub_folder_name, exist_ok=True)

        source_files: List[str] = [f"Grid_Plot_Object_{identity}.png",
                                    f"Grid_Plot_Datapoints_Object_{identity}.png",
                                   f"Hatch_Object_{identity}.png",
                                    f"Divided_Grid_Bounds_Object_{identity}.npy",
                                    f"Grid_Points_Object_{identity}.npy",
                                   f"Hatch_Object_{identity}.npy",
                                   f"CF_Percentage_Object_{identity}.png",
                                   f"CF_Spacing_Object_{identity}.png",
                                   f"Weights_Percentage_Object_{identity}.png"]

        for i, source_file in enumerate(source_files):
            final_name = f"({self.dataset_name})_{source_file}"
            try:
                os.rename(source_file, final_name)
            except FileNotFoundError:
                pass
            source_files[i] = final_name

        for i in range(len(source_files)):

            if clear:
                try:
                    os.remove(source_files[i])
                except FileNotFoundError:
                    pass

            try:
                shutil.move(source_files[i], os.path.join(sub_folder_name, f"{source_files[i]}"))
            except FileNotFoundError:
                pass

    def save_force_connectivity(self, clear: bool = False) -> None:
        """
        Save connectivity data from force grid analysis.

        Parameters
        ----------
        clear : bool, optional
            If True, remove source files instead of moving them (default: False).

        Returns
        -------
        None

        Notes
        -----
        Handles hatch and object connectivity factors and spacing details arrays.
        """

        os.makedirs(self.folder_name, exist_ok=True)

        source_files: List[str] = ["Hatch_Connectivity_Factors.npy",
                                   "Object_Connectivity_Factors.npy",
                                   "Force_Grid_Spacings.npy",
                                   "Spacing_Allocation_Details.npy"]

        for i, source_file in enumerate(source_files):
            final_name = f"({self.dataset_name})_{source_file}"
            try:
                os.rename(source_file, final_name)
            except FileNotFoundError:
                pass
            source_files[i] = final_name

        for i in range(len(source_files)):

            if clear:
                try:
                    os.remove(source_files[i])
                except FileNotFoundError:
                    pass

            try:
                shutil.move(source_files[i], os.path.join(self.folder_name, f"{source_files[i]}"))
            except FileNotFoundError:
                pass

    def save_time_analysis(self, clear: bool = False) -> None:
        """
        Save timing analysis plots and data arrays.

        Parameters
        ----------
        clear : bool, optional
            If True, remove source files instead of moving them (default: False).

        Returns
        -------
        None

        Notes
        -----
        Saves time metrics for grid, boundary, fractal, and connectivity operations.
        """

        sub_folder_name = self.folder_name + f"/Time_Analysis"

        os.makedirs(sub_folder_name, exist_ok=True)

        source_files: List[str] = ["Time_Grid_Spacing.png",
                                   "Time_Grid_Divisions.png",
                                   "Time_Boundary_Spacing.png",
                                   "Time_Boundary_Divisions.png",
                                   "Time_Fractal_Spacing.png",
                                   "Time_Fractal_Divisions.png",
                                   "Time_Connectivity_Spacing.png",
                                   "Time_Connectivity_Divisions.png",
                                   "Time_Force_Grid_Spacing.png",
                                   "Time_Force_Grid_Divisions.png",
                                   "Time_Hatch_Connectivity_Spacing.png",
                                   "Time_Hatch_Connectivity_Divisions.png",
                                   "Time_Force_Grid_Connectivity_Spacing.png",
                                   "Time_Force_Grid_Connectivity_Divisions.png",
                                   "Time_Grid.npy",
                                   "Time_Boundary.npy",
                                   "Time_Fractal.npy",
                                   "Time_Connectivity.npy",
                                   "Time_Force_Grid.npy",
                                   "Time_Hatch_Connectivity.npy",
                                   "Time_Force_Grid_Connectivity.npy",
                                   "X_Grid_Divide.npy",
                                   "X_Grid_Space.npy"]

        for i, source_file in enumerate(source_files):
            final_name = f"({self.dataset_name})_{source_file}"
            try:
                os.rename(source_file, final_name)
            except FileNotFoundError:
                pass

            source_files[i] = final_name

        for i in range(len(source_files)):

            if clear:
                try:
                    os.remove(source_files[i])
                except FileNotFoundError:
                    pass

            try:
                shutil.move(source_files[i], os.path.join(sub_folder_name, f"{source_files[i]}"))
            except FileNotFoundError:
                pass

    def connectivity_deter_save(self, identity, clear: bool = False):
        """
        Save deterioration analysis outputs for a given identity.

        Parameters
        ----------
        identity : int
            Identifier of the structure.
        clear : bool, optional
            If True, remove source files instead of moving them (default: False).

        Returns
        -------
        None

        Notes
        -----
        Saves plots and NumPy arrays related to deterioration analysis.
        """

        sub_folder_name = self.folder_name + f"/Deterioration_Object_{identity}"

        os.makedirs(sub_folder_name, exist_ok=True)

        source_files: List[str] = [f"Time_Length_Object_{identity}.png",
                                   f"Time_Connectivity_Object_{identity}.png",
                                   f"Length_Connectivity_Object_{identity}.png",
                                   f"Iterations_Connectivity_Object_{identity}.png",
                                   f"Timings_Object_{identity}.npy",
                                   f"Lengths_Object_{identity}.npy",
                                   f"CFS_Object_{identity}.npy"]

        for i, source_file in enumerate(source_files):
            final_name = f"({self.dataset_name})_{source_file}"
            try:
                os.rename(source_file, final_name)
            except FileNotFoundError:
                pass
            source_files[i] = final_name

        for i in range(len(source_files)):

            if clear:
                try:
                    os.remove(source_files[i])
                except FileNotFoundError:
                    pass

            try:
                shutil.move(source_files[i], os.path.join(sub_folder_name, f"{source_files[i]}"))
            except FileNotFoundError:
                pass

    def cleanup(self, force_list: List[int]) -> None:
        """
        Remove percentage files for specified force grid objects.

        For each identity in `force_list`, deletes:
        - 'Percent_{identity}.npy': 1D numpy array of coverage percentages per spacing.

        Parameters
        ----------
        force_list : List[int]
            Identities of force grid objects whose percentage files to remove.

        Returns
        -------
        None

        Examples
        --------
        >>> saver = Save('dataset')
        >>> saver.cleanup([1, 2, 3])
        """

        source_files: List[str] = ["Percent"]

        for identity in force_list:
            try:
                os.remove(f"{source_files[0]}_{identity}.npy")
            except FileNotFoundError:
                pass
        
        return None

    def range_force_save(self, identity: int, lower_cap: float, upper_cap: float, step: float, clear: bool = False) -> None:
        """
        Save grid point arrays across spacing ranges for a force grid object.

        Creates files in 'Force_Grid_Object_{identity}/Ranges/':
        - 'Ranges_Grid_Points_Object_{identity}_{cap:.3f}.npy',
          each containing a numpy.ndarray of shape (m, d+1), where
          m is count of points at spacing cap and d is data dimension.

        Parameters
        ----------
        identity : int
            Identifier of the force grid object.
        lower_cap : float
            Lower bound of spacing range (inclusive).
        upper_cap : float
            Upper bound of spacing range (inclusive).
        step : float
            Increment for spacing values between lower_cap and upper_cap.
        clear : bool, optional
            If True, delete source files instead of moving them (default=False).

        Returns
        -------
        None

        Notes
        -----
        Ensures subdirectory 'Force_Grid_Object_{identity}/Ranges' exists.
        """

        sub_folder_name = self.folder_name + f"/Force_Grid_Object_{identity}/Ranges"

        os.makedirs(sub_folder_name, exist_ok=True)

        source_files: List[str] = []

        lower = lower_cap
        higher = lower + step

        while higher <= upper_cap:
            if lower == 0.0:
                source_files.append(f"Grid_Points_Object_{identity}_0.0-{higher}.npy")

            source_files.append(f"Grid_Points_Object_{identity}_{lower}-{higher}.npy")
            lower = higher
            higher = lower + step

        for i, source_file in enumerate(source_files):
            final_name = f"({self.dataset_name})_{source_file}"
            try:
                os.rename(source_file, final_name)
            except FileNotFoundError:
                pass
            source_files[i] = final_name

        for i in range(len(source_files)):

            if clear:
                try:
                    os.remove(source_files[i])
                except FileNotFoundError:
                    pass

            try:
                shutil.move(source_files[i], os.path.join(sub_folder_name, f"{source_files[i]}"))
            except FileNotFoundError:
                pass

        return None
