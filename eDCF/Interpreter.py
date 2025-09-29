import numpy as np
from typing import List
from tzlocal import get_localzone
from datetime import datetime


class Interpreter:
    """
    Generate 'Report.txt' summarizing analyses not captured by graphs.

    Compiles timing, connectivity, fractal, topological, and algorithmic results.

    Attributes
    ----------
    structures : List[str]
        Descriptors of data sets included in report.
    algorithm : str
        Algorithm name used for grid labeling.

    Methods
    -------
    report(method: List[bool]) -> None
        Create or overwrite the main report file.
    add_force_info(identity_list: List[int], basis: float=50.0, spacing_allocator_ctrl: bool=False) -> None
        Append force grid connectivity details.
    add_topological_info(force_identity: List[int]) -> None
        Append topological dimension details.
    add_fractal_obj_info(force_identity: List[int]) -> None
        Append fractal dimension details.
    """

    def __init__(self, structures: List[str], algorithm: str = ""):
        """
        Initialize interpreter for report generation.

        Parameters
        ----------
        structures : List[str]
            List of structure descriptions to include in report.
        algorithm : str, optional
            Algorithm name for grid labeling (default: '').

        Examples
        --------
        >>> interp = Interpreter(['Dataset1', 'Dataset2'], 'SVM')
        """

        self.structures = structures
        self.algorithm = algorithm

    def report(self, method: List[bool]) -> None:
        """
        Write main report content to 'Report.txt'.

        Parameters
        ----------
        method : List[bool]
            Flags for boundary calculation method:
            False -> N Cube Neighbour, True -> Neighbour in Set.

        Returns
        -------
        None

        Notes
        -----
        Expects files:
        - Grid_Info.npy
        - algorithms/Testing_Score.npy
        - algorithms/Hyper_Param.npy
        - Boundary_Type.npy
        - Dimensions.npy
        - Connectivity_Factors.npy
        """

        file = open("Report.txt", 'w')

        now = datetime.now(get_localzone())

        str_time = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")

        file.write(str_time + "\n\n")

        file.write("For the Datapoints stored in Datapoints.npy file, the following analysis:\n\n")

        file.write(f"\t-> Structures: \n")

        for i, struct in enumerate(self.structures):

            file.write(f"\t\t->Structure Number {i + 1}: {struct}\n")

        file.write("\n\t-> Calculated Labelled Grid Points in the file Grid_Points.npy with the following Grid Parameters:\n")

        info = np.load("Grid_Info.npy")
        testing_score = np.load("algorithms/Testing_Score.npy")
        file.write(f"\t\t-> Grid Spacing: {info[0]}\n")
        file.write(f"\t\t-> Grid Boundary Buffer Value: {info[1]}, (means the amount the grid has been extended to prevent loss of information)\n")
        file.write(f"\t\t-> Grid Division into Mini Grids: {int(info[2]) ** int(info[3])}\n")
        file.write(f"\t\t-> Grid Dimension: {int(info[3])}\n")
        file.write(f"\t\t-> Algorithm used to calculate Grid Labels: {self.algorithm} with Testing Score: {testing_score}\n")

        hyper_params = np.load("algorithms/Hyper_Param.npy", allow_pickle=True).item()

        keys = list(hyper_params.keys())

        file.write("\t\t\t-> Hyper Parameters used for the Algorithm:\n")

        for key in keys:
            file.write(f"\t\t\t\t-> {key}: {hyper_params[key]}:\n")

        if not method[0]:
            file.write("\n\t-> Calculated Decision Boundary using N Cube Neighbour Method in the file Neighbour_Boundary.npy\n\n")
        else:
            file.write("\n\t-> Calculated Decision Boundary using Neighbour in Set Method in the file Neighbour_Boundary.npy\n\n")

        file.write("\t-> Calculated Fractal Dimension of the Boundaries in Dimensions.npy and the Boundary Type(between which two structures) in Boundary_Type.npy\n")
        file.write("\t-> Calculated Connectivity Factor of the Boundaries in Connectivity_Factors.npy\n")

        boundary_type = np.load("Boundary_Type.npy")
        dimensions = np.load("Dimensions.npy")
        connectivity_factors = np.load("Connectivity_Factors.npy")

        for i, bound_type in enumerate(boundary_type):

            low = bound_type[0]
            high = bound_type[1]

            if dimensions[i] == -1:
                file.write(f"\n\t\t-> Boundary between {low} and {high} does not exist\n")
            else:
                file.write(f"\n\t\t-> Fractal Dimension for Boundary between {low} and {high}: {dimensions[i]}\n")
                file.write(f"\t\t-> Connectivity Factor for Boundary between {low} and {high}: {connectivity_factors[i]}\n")

        file.write("\n")
        file.close()

    @staticmethod
    def add_force_info(identity_list: List[int], basis: float = 50.0, spacing_allocator_ctrl: bool = False) -> None:
        """
        Append force grid connectivity details to 'Report.txt'.

        Parameters
        ----------
        identity_list : List[int]
            Identifiers of force grid objects.
        basis : float, optional
            Percentage basis for estimation (default: 50.0).
        spacing_allocator_ctrl : bool, optional
            Indicates if spacing-range allocation was used.

        Returns
        -------
        None

        Notes
        -----
        Reads Hatch_Connectivity_Factors.npy, Object_Connectivity_Factors.npy,
        Force_Grid_Spacings.npy, Analysis.npy or Spacing_Allocation_Details.npy.
        """

        file = open("Report.txt", 'a')

        file.write(f"\t-> Connectivity Factor for Objects would be stored in Object_Connectivity_Factors.npy and for Hatches it would be stored in Hatch_Connectivity_Factors.npy\n")

        spacings = np.load("Force_Grid_Spacings.npy")

        try:

            cfs_h = np.load("Hatch_Connectivity_Factors.npy")
            file.write(f"\t-> Hatch Connectivity Factors: \n")
            for i, identity in enumerate(identity_list):
                file.write(f"\t\t-> Object Hatch {identity} - {cfs_h[i]} ({spacings[i]})\n")

        except FileNotFoundError:
            pass


        try:

            analysis = np.load("Analysis.npy", allow_pickle=True)

            file.write(f"\n\t-> NOTE: These values are calculated/estimated for a basis {basis}% from a Polynomial Estimation for CF and Piecewise Polynomial Estimation for ln(Spacing) from calculated CF.\n")

        except FileNotFoundError:
            if spacing_allocator_ctrl:
                spacing_allocation_details = np.load("Spacing_Allocation_Details.npy")
                file.write(
                    f"\n\t-> NOTE: These values are calculated while the Range of Percentage of Total Set of Points was {spacing_allocation_details[2]}% and {spacing_allocation_details[3]}%.\n")

        try:

            cfs_o = np.load("Object_Connectivity_Factors.npy")
            file.write(f"\n\t-> Object Connectivity Factors: \n")
            for i, identity in enumerate(identity_list):
                file.write(f"\t\t-> Object {identity} - {cfs_o[i]} (Spacing Value - {spacings[i, int(spacings[i, 2])]})\n")

        except FileNotFoundError:
            pass

        file.close()

    @staticmethod
    def add_topological_info(force_identity: List[int]):
        """
        Append topological dimension details to 'Report.txt'.

        Parameters
        ----------
        force_identity : List[int]
            Identifiers of force grid objects for which to add info.

        Returns
        -------
        None

        Notes
        -----
        Reads Topological_Dimensions_Boundaries.npy,
        Weighted_Topological_Dimensions_Boundaries.npy,
        Weights_Boundaries_Normalized.npy, Topological_Dimensions_Object.npy,
        Weighted_Topological_Dimensions_Force.npy, Weights_Force_Normalized.npy,
        Topological_Dimensions_Hatch.npy, Weighted_Topological_Dimensions_Hatch.npy,
        Weights_Hatch_Normalized.npy.
        """

        file = open("Report.txt", 'a')

        try:
            td_boundaries = np.load("Topological_Dimensions_Boundaries.npy")
            td_boundaries_weighted = np.load("Weighted_Topological_Dimensions_Boundaries.npy")
            weights_boundaries_norm = np.load("Weights_Boundaries_Normalized.npy")
            file.write(f"\n\t-> Topological Dimensions for Boundaries is stored in Topological_Dimensions_Boundaries.npy file\n\n")

            boundary_type = np.load("Boundary_Type.npy")
            dimensions = np.load("Dimensions.npy")

            for i, bound_type in enumerate(boundary_type):

                low = bound_type[0]
                high = bound_type[1]

                if dimensions[i] == -1:
                    file.write(f"\t\t-> Boundary between {low} and {high} does not exist\n")
                else:
                    file.write(f"\t\t-> Topological Dimension for Boundary between {low} and {high}(CF Based): {td_boundaries[i]}\n\n")
                    file.write(f"\t\t-> Topological Dimension for Boundary between {low} and {high}(Weight Based): {td_boundaries_weighted[i]}\n")
                    file.write(f"\t\t\t-> Weights for Boundary {low} - {high}: \n")

                    for j in range(len(weights_boundaries_norm[i])):

                        file.write(f"\t\t\t\t->Topology {j} = {weights_boundaries_norm[i, j]}\n")

                file.write("\n")

        except FileNotFoundError:
            pass

        try:
            td_force = np.load("Topological_Dimensions_Object.npy")
            td_force_weighted = np.load("Weighted_Topological_Dimensions_Force.npy")
            weights_force_norm = np.load("Weights_Force_Normalized.npy")
            file.write(f"\n\t-> Topological Dimensions for Force Grid Objects is stored in Topological_Dimensions_Object.npy file\n\n")

            for i, identity in enumerate(force_identity):

                file.write(f"\t\t-> Topological Dimension for Object {identity}(CF Based): {td_force[i]}\n\n")
                file.write(f"\t\t-> Topological Dimension for Object {identity}(Weight Based): {td_force_weighted[i]}\n")

                file.write(f"\t\t\t-> Weights for Object {identity}: \n")

                for j in range(len(weights_force_norm[i])):
                    file.write(f"\t\t\t\t->Topology {j} = {weights_force_norm[i, j]}\n")

                file.write("\n")

        except FileNotFoundError:
            pass

        try:
            td_hatch = np.load("Topological_Dimensions_Hatch.npy")
            td_hatch_weighted = np.load("Weighted_Topological_Dimensions_Hatch.npy")
            weights_hatch_norm = np.load("Weights_Hatch_Normalized.npy")
            file.write(f"\n\t-> Topological Dimensions for Force Grid Hatches is stored in Topological_Dimensions_Hatch.npy file\n\n")

            for i, identity in enumerate(force_identity):

                file.write(f"\t\t-> Topological Dimension for Hatch {identity}(CF based): {td_hatch[i]}\n\n")

                file.write(f"\t\t-> Topological Dimension for Hatch {identity}(Weight Based): {td_hatch_weighted[i]}\n")
                file.write(f"\t\t\t-> Weights for Hatch {identity}: \n")

                for j in range(len(weights_hatch_norm[i])):
                    file.write(f"\t\t\t\t->Topology {j} = {weights_hatch_norm[i, j]}\n")

                file.write("\n")

        except FileNotFoundError:
            pass

        file.close()

    @staticmethod
    def add_fractal_obj_info(force_identity: List[int]):
        """
        Append fractal dimension details to 'Report.txt'.

        Parameters
        ----------
        force_identity : List[int]
            Identifiers of force grid objects.

        Returns
        -------
        None

        Notes
        -----
        Reads Fractal_Dimensions_Objects.npy.
        """

        file = open("Report.txt", 'a')

        try:
            fd_force = np.load("Fractal_Dimensions_Objects.npy")
            file.write(f"\n\t-> Fractal Dimensions for Force Grid Objects is stored in Fractal_Dimensions_Object.npy file\n\n")

            for i, identity in enumerate(force_identity):

                file.write(f"\t\t-> Fractal Dimension for Object {identity}: {fd_force[i]}\n")

                file.write("\n")

        except FileNotFoundError:
            pass

        file.close()