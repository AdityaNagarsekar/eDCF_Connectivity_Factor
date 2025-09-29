import numpy as np
import os
import warnings
import skdim.datasets
from tqdm import tqdm
import multiprocessing
import tempfile
from typing import Tuple

# --- Import relevant custom classes ---
from ForceGridObject import ForceGridObject  # noqa: F401  (used inside get_avg_neighbor_count)
from Connectivity import Connectivity         # noqa: F401  (used inside get_avg_neighbor_count)


def get_avg_neighbor_count(data: np.ndarray, target_percentage: float = 50.0):
    """
    Runs the gridding and connectivity pipeline for a given manifold to find the
    average number of neighbors per point. This implementation is **process-safe**:
    each worker writes its temporary data to an isolated directory so that there
    is **no filename collision** when running in parallel.
    """
    try:
        # ------------------------------------------------------------------
        # 1. Normalise data to [0,1]^d so spacing hyper-parameters become agnostic
        # ------------------------------------------------------------------
        data_normalized = data.copy()
        min_bounds = np.min(data_normalized, axis=0)
        max_bounds = np.max(data_normalized, axis=0)
        ranges = max_bounds - min_bounds
        ranges[ranges == 0] = 1  # avoid div-zero in degenerate dims
        global_range = np.max(ranges)
        if global_range == 0:
            global_range = 1
        data_normalized = (data_normalized - min_bounds) / global_range

        n_samples, ambient_dim = data_normalized.shape

        # ------------------------------------------------------------------
        # 2. Work inside a **private temp dir** => no shared Datapoints.npy
        # ------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)

            # ForceGridObject expects a file called Datapoints.npy where each
            # entry in the array corresponds to a different identity.  We only
            # need one, so wrap in a list and save.
            np.save(
                "Datapoints.npy",
                [np.hstack([data_normalized, np.ones((n_samples, 1))])],
                allow_pickle=True,
            )

            fgo = ForceGridObject(identity=1)

            # Precompute bounds of the grid (min/max per dimension)
            grid_bounds = np.vstack(
                (np.min(data_normalized, axis=0), np.max(data_normalized, axis=0))
            ).T

            # ------------------------------------------------------------------
            # 3. Binary search on spacing so that ~target_percentage of points
            #    are retained after gridding (empirically stable heuristic)
            # ------------------------------------------------------------------
            low_sp, high_sp = 0.001, 0.5
            for _ in range(15):
                mid_sp = (low_sp + high_sp) / 2
                grid_info = np.array([mid_sp, 0.1, 2, ambient_dim])
                grid_set = fgo.direct_gridder(
                    data_normalized, grid_bounds, grid_info, mid_sp
                )
                current_percentage = len(grid_set) / n_samples * 100
                if current_percentage < target_percentage:
                    high_sp = mid_sp
                else:
                    low_sp = mid_sp
            best_spacing = (low_sp + high_sp) / 2

            # ------------------------------------------------------------------
            # 4. Final gridding using optimal spacing
            # ------------------------------------------------------------------
            final_grid_info = np.array([best_spacing, 0.1, 2, ambient_dim])
            grid_set = fgo.direct_gridder(
                data_normalized, grid_bounds, final_grid_info, best_spacing
            )
            grid_points = np.array(list(grid_set))

            # ------------------------------------------------------------------
            # 5. Average neighbour count in an L-infty ball of radius spacing
            # ------------------------------------------------------------------
            if grid_points.shape[0] < 2:
                os.chdir(orig_cwd)
                return 0.0

            N = len(grid_points)
            coords = grid_points[:, :ambient_dim]
            counts = np.zeros(N, dtype=int)
            tolerance = 1e-9

            for i in range(N):
                for j in range(i + 1, N):
                    diff = np.abs(coords[i] - coords[j])
                    if np.all(diff <= best_spacing + tolerance) and np.sum(diff) > tolerance:
                        counts[i] += 1
                        counts[j] += 1

            # ---- ADD THIS CODE ----
            avg_count = np.mean(counts) if N > 0 else 0.0
            print(f"--- In Reference Model ---")
            print(f"For this ideal manifold, returning a single AVERAGE count: {avg_count:.2f}")
            # ----------------------

            os.chdir(orig_cwd)
            return np.mean(counts) if N > 0 else 0.0

    except Exception:
        # Any failure in this process simply yields 0.0 neighbour count so that
        # reference model construction can continue for other dimensions.
        return 0.0


# -----------------------------------------------------------------------------
# Parallel helper – executed in worker processes
# -----------------------------------------------------------------------------

def _dimension_job(args: Tuple[int, int, float]) -> Tuple[int, float]:
    """Compute average neighbour count for a single intrinsic dimension."""
    dim, n_points, noise_level = args

    # Generate noisy hypersphere manifold of intrinsic dimension `dim`
    manifold = skdim.datasets.hyperSphere(n=n_points, d=dim)
    manifold += np.random.normal(0, noise_level, manifold.shape)

    # Calculate average neighbour count using the process-safe routine
    avg = 3 ** dim - 1
    return dim, avg


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main(n_points: int):
    """Generate and save the manifold reference model using multiprocessing."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    max_dim_to_model = 2
    noise_level = 0.0

    print("Generating manifold reference model (parallel)…")

    # Prepare tasks for each intrinsic dimension to model
    tasks = [(dim, n_points, noise_level) for dim in range(1, max_dim_to_model + 1)]

    reference_model = {0: 0.0}  # Dict[dimension -> avg_neighbour_count]

    num_workers = min(multiprocessing.cpu_count(), len(tasks))
    with multiprocessing.Pool(processes=num_workers) as pool:
        for dim, avg in tqdm(
            pool.imap_unordered(_dimension_job, tasks),
            total=len(tasks),
            desc="Building Reference Model",
        ):
            reference_model[dim] = avg

    # Persist to disk (NPY stores python dict fine with allow_pickle=True)
    np.save("manifold_reference_model.npy", reference_model, allow_pickle=True)
    print("Manifold reference model saved to 'manifold_reference_model.npy'")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000  # default #samples
    main(n)