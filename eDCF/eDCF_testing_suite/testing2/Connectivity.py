import math
import multiprocessing
import psutil
import numpy as np
from typing import Any
from tqdm import tqdm
from joblib import Parallel, delayed

from NCubeNeighbours import NCubeNeighbours
from Weight import Weight


class Connectivity:
    """Compute connectivity factor for spatial point sets on a grid."""

    # ---------------------------------------------------------------------
    # Helper: decimal-place counter (unchanged)
    # ---------------------------------------------------------------------
    @staticmethod
    def count_decimal_places(number, max_to_consider_trailing: int = 8):
        try:
            s_number = str(number)
            if 'e' in s_number:
                return int(s_number.split('e-')[1])
            if '.' in s_number:
                return len(s_number.split('.')[1])
        except Exception:
            return max_to_consider_trailing
        return 0

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(self, points: np.ndarray, spacing: float, neighbour_set_method: bool = False):
        self.spacing = spacing
        self.precision = Connectivity.count_decimal_places(spacing)
        self.dimension = len(points[0, :-1])
        self.neighbour_set_method = neighbour_set_method

        if neighbour_set_method:
            self.points = (np.round(points[:, :-1], decimals=self.precision) * (10 ** self.precision)).astype(int)
        else:
            # store as tuples for O(1) membership checks
            self.points_tup = {tuple(round(coord, self.precision) for coord in p) for p in points}

    # ---------------------------------------------------------------------
    # Lookup‑based method (sequential; relies on stateful NCubeNeighbours)
    # ---------------------------------------------------------------------
    def calculate_connectivity_factor(self, points: np.ndarray):
        """Original lookup‑based connectivity (left as single‑thread for safety)."""
        max_model_dim = 2
        wt = Weight(dim=max_model_dim)

        n_cube = NCubeNeighbours(self.dimension, points[0])
        neighbours = []

        for point in tqdm(points, desc="Lookup Connectivity", leave=False):
            point_tuple = tuple(round(coord, self.precision) for coord in point)
            current_center_tuple = tuple(round(c, self.precision) for c in n_cube._NCubeNeighbours__center)
            if not np.allclose(point_tuple, current_center_tuple):
                n_cube = NCubeNeighbours(self.dimension, np.array(point_tuple))
                neighbours = n_cube.get_ndcube_neighbours(self.spacing)
            elif not neighbours:
                neighbours = n_cube.get_ndcube_neighbours(self.spacing)

            neighbours_set = {tuple(round(coord, self.precision) for coord in n) for n in neighbours}
            common_points = neighbours_set.intersection(self.points_tup)
            wt.add_point_info(len(common_points))

        return -1.0, len(points), wt  # dummy connectivity factor, plus Weight

    # ---------------------------------------------------------------------
    # Pair‑wise method – now thread‑parallel
    # ---------------------------------------------------------------------
    @staticmethod
    def _calculate_pairwise(points: np.ndarray, dimension: int, spacing: float):
        """Pairwise comparison with thread‑level parallelism via joblib."""
        N = len(points)
        d = dimension
        wt = Weight(dim=20)
        if N < 2:
            return 0.0, wt

        coords = points[:, :d]
        tol = 1e-9

        # per‑index neighbour counter (broadcasted difference)
        def _count_for(i: int) -> int:
            diff = np.abs(coords[i] - coords)  # shape (N,d)
            within = np.all(diff <= spacing + tol, axis=1)
            return int(np.sum(within)) - 1  # exclude self

        # Limit threads to avoid over‑subscription when used inside a process pool
        n_threads = max(1, min(4, multiprocessing.cpu_count() // 2))
        counts = Parallel(n_jobs=n_threads, backend="threading", prefer="threads")(
            delayed(_count_for)(i) for i in range(N)
        )

        # ---- ADD THIS CODE ----
        print(f"\n--- In Main eDCF Logic ---")
        print(f"List of {len(counts)} absolute neighbor counts (first 20): {counts[:20]}")
        print(f"The average is {np.mean(counts):.2f}, but it is NOT used directly.")
        # ----------------------

        for c in counts:
            wt.add_point_info(c)

        return -1.0, wt  # dummy CF, plus Weight

    # ---------------------------------------------------------------------
    # Public dispatcher
    # ---------------------------------------------------------------------
    @staticmethod
    def calc_connectivity_general(points: list, spacing: float, neighbour_set_method: bool = False):
        if not points:
            return -1, None
        points = [p for p in points if p.shape[0] > 0]
        if not points:
            return -1, None

        points_arr = np.vstack(points)
        points_arr = np.array(list({tuple(pt) for pt in points_arr}), dtype=float)
        if points_arr.size == 0:
            return -1, None

        d = points_arr.shape[1] - 1
        use_pairwise = d > 3

        if use_pairwise:
            _, wt = Connectivity._calculate_pairwise(points_arr, d, spacing)
            return -1.0, wt
        else:
            conn = Connectivity(points_arr, spacing, neighbour_set_method)
            _, _, wt = conn.calculate_connectivity_factor(points_arr)
            return -1.0, wt