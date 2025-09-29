# import statements
import traceback
import warnings

import numpy as np
import multiprocessing

import psutil
from scipy.optimize import curve_fit

from BoundaryExtractor import BoundaryExtractor


class FractalDetector:
    """
    Bulk detection and fractal dimension analysis for boundary systems.

    Uses boundaries from `BoundaryExtractor` to compute various fractal dimensions
    and classify boundary types. Primary output files:
    - 'Dimensions.npy': array of fractal dimension estimates per boundary type.
    - 'Boundary_Type.npy': array of structure ID pairs for each boundary.

    Methods
    -------
    __init__() -> None
        Load grid bounds, datapoint counts, and grid parameters from .npy files.
    compute_fractal_dimension(data_points: numpy.ndarray) -> float
        Box-counting dimension for a single boundary dataset.
    linear_func(x, slope, intercept) -> numpy.ndarray
        Linear function for regression.
    compute_nd_fractal_dimension(coordinates: numpy.ndarray,
                                 num_scales: int = 12,
                                 min_segment_points: int = 5) -> float
        N-dimensional box-counting dimension with gliding boxes.
    compute_nd_fractal_dimension_batched(data_points: numpy.ndarray,
                                        max_points_per_batch: int = 10000) -> float
        Batched ND fractal dimension for large datasets.
    compute_hausdorff_dimension_enhanced(data_points: numpy.ndarray,
                                        max_sample_size: int = 5000,
                                        min_radius_factor: float = 0.001,
                                        max_radius_factor: float = 0.5,
                                        num_radii: int = 50,
                                        num_trials: int = 5,
                                        robust: bool = True,
                                        debug: bool = False) -> float
        High-precision Hausdorff dimension estimation with confidence intervals.
    get_available_cores(threshold: int = 50) -> int
        Available CPU cores based on system usage.
    detect(create_boundary_type_only: bool = False) -> None
        Generate 'Dimensions.npy' and 'Boundary_Type.npy' using multiprocessing.
    detect_general(points: numpy.ndarray) -> float
        Compute single-dataset fractal dimension (ND or Hausdorff based on `self.num`).

    Examples
    --------
    >>> detector = FractalDetector()
    >>> detector.detect()  # creates Dimensions.npy and Boundary_Type.npy
    >>> dims = np.load('Dimensions.npy')
    >>> types = np.load('Boundary_Type.npy')
    """

    def __init__(self):
        """
        Initialize detector by loading grid and data information.

        Attributes
        ----------
        grid_bounds : numpy.ndarray, shape (n_dims, 2)
            Coordinate bounds loaded from 'Grid_Bounds.npy'.
        num_structs : int
            Number of structures, based on 'Datapoints.npy'.
        num : int
            Total count of data points across all structures.
        grid_spacing : float
            Grid spacing value from 'Grid_Info.npy'.
        boundary_buffer : float
            Buffer distance for boundary calculation from 'Grid_Info.npy'.
        grid_divide : int
            Number of grid divisions from 'Grid_Info.npy'.
        grid_dimension : int
            Dimensionality of the grid from 'Grid_Info.npy'.

        Notes
        -----
        Requires files: 'Datapoints.npy', 'Grid_Info.npy', 'Grid_Bounds.npy'.
        """

        self.__grid_bounds = np.load("Grid_Bounds.npy", allow_pickle=True)  # Getting Grid Boundaries for whole grid.
        self.__num_structs: int = len(np.load("Datapoints.npy", allow_pickle=True))  # loading datapoints to get the number of structures present.

        self.__num = 0
        datapoints_list = np.load("Datapoints.npy", allow_pickle=True)

        for datapoints in datapoints_list:
            self.__num += len(datapoints)

        # Unpacking "Grid_Info.npy"
        info = np.load("Grid_Info.npy")  # loading grid info
        self.__grid_spacing = info[0]
        self.__boundary_buffer = info[1]
        self.__grid_divide = int(info[2])
        self.__grid_dimension = int(info[3])

    # Function to compute the fractal dimension using the improved box-counting method
    def compute_fractal_dimension(self, data_points: np.ndarray) -> float:
        """
        Compute box-counting fractal dimension for boundary point set.

        Parameters
        ----------
        data_points : numpy.ndarray, shape (n_points, n_dims)
            2D array of boundary coordinates.

        Returns
        -------
        float
            Estimated fractal dimension (slope of log(N) vs log(1/scale)).
            Returns -1.0 if input is invalid or an error occurs.

        Notes
        -----
        - Uses `self.__grid_spacing` as the minimum box size.
        - Scales are powers of two up to half the data range.
        - Uses overlapping boxes with shifts of 0 and scale/2 for count minimization.
        - Linear regression applied to `log(1/scale)` vs `log(N)`.

        Examples
        --------
        >>> detector = FractalDetector()
        >>> dim = detector.compute_fractal_dimension(boundary_points)
        """

        try:
            # Ensure data_points is a proper 2D numpy array
            if not isinstance(data_points, np.ndarray):
                data_points = np.array(data_points, dtype=float)

            # Handle dimensionality issues
            if data_points.ndim != 2:
                if data_points.ndim == 1:
                    data_points = data_points.reshape(-1, 1)
                else:
                    return -1.0

            # The original code uses std_dev and mean_std but doesn't actually use them
            # in the rest of the calculation, so we can safely remove these lines:
            # std_dev = np.std(data_points, axis=0)
            # mean_std = np.mean(std_dev)

            # Adjust the scales based on the data's variability
            min_scale = self.__grid_spacing  # Starting from the grid spacing
            max_scale = np.max(self.__grid_bounds[:, 1] - self.__grid_bounds[:, 0])  # Max range in any dimension

            # Determine number of scales based on the data range and min_scale
            num_scales = int(np.floor(np.log2(max_scale / min_scale)))
            scales = min_scale * 2 ** np.arange(0, num_scales)
            scales = scales[scales <= max_scale / 2]  # Ensure scales are not larger than half the data range

            Ns = []
            log_scales = []
            for scale in scales:
                # Partition the space into boxes of size 'scale'
                # Use overlapping boxes (Modification C)
                counts = []
                shifts = [0, scale / 2]
                for shift in shifts:
                    # Compute the edges of the grid
                    bin_edges = []
                    for dim in range(self.__grid_dimension):
                        min_bound = self.__grid_bounds[dim, 0] - self.__boundary_buffer
                        max_bound = self.__grid_bounds[dim, 1] + self.__boundary_buffer
                        bins = np.arange(min_bound + shift, max_bound + scale, scale)
                        bin_edges.append(bins)
                    # Use np.histogramdd to count the number of occupied boxes
                    H, edges = np.histogramdd(data_points, bins=bin_edges)
                    N = np.sum(H > 0)
                    counts.append(N)
                # Take the minimum number of boxes (Modification B)
                N = min(counts)
                Ns.append(N)
                log_scales.append(np.log(1 / scale))

            log_Ns = np.log(Ns)
            log_scales = np.array(log_scales)

            # Perform linear regression to find the slope
            def linear_func(x, a, b):
                return a * x + b

            popt, pcov = curve_fit(linear_func, log_scales, log_Ns)
            fractal_dimension: float = popt[0]
            return fractal_dimension
        except Exception as e:
            print(f"Error computing fractal dimension: {e}")
            return -1.0

    def linear_func(self, x, slope, intercept):
        """
        Linear model function: computes y = slope * x + intercept.

        Parameters
        ----------
        x : array_like or float
            Independent variable values.
        slope : float
            Slope of the linear function.
        intercept : float
            Intercept of the linear function.

        Returns
        -------
        numpy.ndarray or float
            Result of linear model applied to x.
        """

        return slope * x + intercept

    def compute_nd_fractal_dimension(self, coordinates: np.ndarray, num_scales: int = 12, min_segment_points: int=5) -> float:
        """
        Calculates the box-counting dimension by finding the most linear
        region of the log-log plot automatically.

        Uses a memory-optimized approach with a shifting grid ("gliding box").

            Assumes input is coordinate data only.

            :param coordinates: n-dimensional numpy array with format [N_points, N_dimensions]
                                containing only the coordinates of the points.
            :param num_scales: Approximate number of different box sizes (scales) to use.
                               More scales can improve accuracy but increase computation time.
                               (Default: 20)
            :param min_segment_points: The minimum number of data points required in the
                                       log-log plot segment to consider for linear fitting.
                                       (Default: 5)
            :return: Estimated box counting dimension as a float. Returns np.nan if
                     calculation fails (e.g., insufficient points, memory error,
                     all points identical, no suitable linear segment found).
            """
        # --- Input Validation and Preparation (mostly unchanged) ---
        # print(coordinates) # Keep if you need to debug the input data

        if not isinstance(coordinates, np.ndarray):
            try:
                coordinates = np.array(coordinates, dtype=np.float32)
            except Exception as e:
                print(f"Error converting input to numpy array: {e}")
                return np.nan
        elif coordinates.dtype != np.float32:
            coordinates = coordinates.astype(np.float32, copy=False)

        if coordinates.ndim != 2 or coordinates.shape[0] < 2 or coordinates.shape[1] == 0:
            print("Warning: Input data must be a 2D array with at least 2 points.")
            return np.nan

        try:
            data_min = np.min(coordinates, axis=0)
            data_max = np.max(coordinates, axis=0)
            max_extent = np.max(data_max - data_min) + 1e-20  # Use a small epsilon

            if max_extent <= 1e-20:
                print("Warning: All points seem identical or data range is zero. Returning dimension 0.")
                return 0.0

            # --- Scale Determination (mostly unchanged) ---
            min_log_scale = np.log2(max_extent / (2 ** (num_scales)))
            # Adjust max_scale slightly - often the largest scales deviate significantly
            # Let's try limiting it a bit more aggressively, e.g., max_extent / 4
            max_log_scale = np.log2(max_extent / 4)  # ADJUSTED UPPER LIMIT

            if min_log_scale >= max_log_scale:
                # If range is tiny or num_scales too high, try a different range
                min_log_scale = np.log2(max_extent / (2 ** 5))  # Lower limit based on 5 steps down
                max_log_scale = np.log2(max_extent / 2)  # Upper limit less aggressive
                if min_log_scale >= max_log_scale:  # Still bad?
                    print("Warning: Cannot determine a valid scale range. Check data extent and num_scales.")
                    return np.nan  # Fallback

            scales = np.logspace(min_log_scale, max_log_scale, num=num_scales, base=2.0)
            scales = np.unique(scales[scales > 1e-20])  # Ensure positive and unique

            if len(scales) < min_segment_points:  # Need enough potential scales
                print(f"Warning: Could not determine enough distinct scales ({len(scales)}) "
                      f"to find a linear segment of length {min_segment_points}.")
                return np.nan

            # --- Box Counting Loop (unchanged) ---
            Ns = []
            valid_scales = []
            buffer = 1e-15  # Slightly larger buffer might help with edge cases

            for scale in scales:
                min_count_for_scale = float('inf')
                for shift_factor in [0.0, 0.5]:  # Use floats
                    shift = shift_factor * scale
                    bin_edges = []
                    try:
                        for dim in range(coordinates.shape[1]):
                            dim_min = data_min[dim] - buffer
                            dim_max = data_max[dim] + buffer
                            # Ensure the range fully covers data + shift + scale
                            # Use ceil to ensure the upper limit is definitely past max
                            num_bins = int(np.ceil((dim_max - dim_min) / scale)) + 2  # Generous bin count
                            start = dim_min + shift
                            edges = start + np.arange(num_bins + 1) * scale
                            # Refine edges to tightly bracket the data range +/- buffer
                            valid_idx = (edges >= dim_min - scale) & (edges <= dim_max + scale)
                            refined_edges = edges[valid_idx]
                            if len(refined_edges) < 2:  # Need at least one bin
                                refined_edges = np.array([start, start + scale])  # Fallback bin

                            # Ensure edges are strictly increasing
                            diffs = np.diff(refined_edges)
                            if not np.all(diffs > 1e-15):  # Allow for tiny float variations
                                # If edges are too close or decreasing, try simpler range
                                refined_edges = np.arange(dim_min + shift, dim_max + shift + scale, scale)
                                if not np.all(np.diff(refined_edges) > 1e-15):
                                    # print(f"Debug: Problematic edges for dim {dim}, scale {scale}")
                                    raise ValueError("Bin edges are not strictly increasing")  # Force skip
                            bin_edges.append(refined_edges)

                        # Check if bin_edges construction failed for any dimension
                        if len(bin_edges) != coordinates.shape[1]:
                            continue  # Skip this shift if edge creation failed

                        H, _ = np.histogramdd(coordinates, bins=bin_edges)
                        count = np.sum(H > 0)
                        min_count_for_scale = min(min_count_for_scale, count)
                        del H
                    except MemoryError:
                        print(f"Warning: MemoryError during histogram calculation for scale {scale}. Skipping scale.")
                        min_count_for_scale = 0
                        break
                    except ValueError as ve:
                        # print(f"Warning: ValueError during histogram calculation for scale {scale}: {ve}. Skipping shift.")
                        # Let loop find minimum, don't reset min_count_for_scale here, just skip this invalid shift attempt
                        continue  # Try next shift factor or scale
                    except Exception as e:  # Catch other unexpected histogram errors
                        print(f"Warning: Unexpected error during histogram for scale {scale}: {e}. Skipping scale.")
                        min_count_for_scale = 0
                        break

                if min_count_for_scale > 0 and np.isfinite(min_count_for_scale):
                    Ns.append(min_count_for_scale)
                    valid_scales.append(scale)  # Store the scale used for this Ns

            # --- Automatic Linear Segment Finding ---
            Ns = np.array(Ns)
            valid_scales_arr = np.array(valid_scales)

            if len(Ns) < min_segment_points:  # Check if enough points survived box counting
                print(f"Warning: Insufficient valid points ({len(Ns)}) after box counting "
                      f"to find a linear segment of length {min_segment_points}.")
                return np.nan

            log_Ns = np.log(Ns)
            # Use the valid scales corresponding to the Ns counts
            log_scales_inv = np.log(1.0 / valid_scales_arr)

            # --- Sliding Window ---
            best_r_squared = -1.0
            best_segment_indices = None
            best_popt = None
            best_pcov = None

            print(f"Searching for best linear segment (min length {min_segment_points}) among {len(log_Ns)} points...")

            for i in range(len(log_Ns) - min_segment_points + 1):
                for j in range(i + min_segment_points, len(log_Ns) + 1):
                    # Extract segment
                    x_segment = log_scales_inv[i:j]
                    y_segment = log_Ns[i:j]

                    # Check for constant values which break curve_fit variance check
                    if len(np.unique(x_segment)) < 2 or len(np.unique(y_segment)) < 2:
                        continue

                    try:
                        # Perform fit on the segment
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignore potential overflows
                            warnings.simplefilter("ignore",
                                                  category=FutureWarning)  # Ignore potential future numpy warnings
                            popt_segment, pcov_segment = curve_fit(self.linear_func, x_segment, y_segment, check_finite=True)

                        # Calculate R-squared for this segment's fit
                        residuals = y_segment - self.linear_func(x_segment, *popt_segment)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((y_segment - np.mean(y_segment)) ** 2)

                        if ss_tot <= 1e-10:  # Avoid division by zero if y_segment is constant
                            r_squared = 1.0 if ss_res <= 1e-10 else 0.0
                        else:
                            r_squared = 1.0 - (ss_res / ss_tot)

                        # Check if this is the best fit so far
                        if r_squared > best_r_squared:
                            # Check for valid covariance before accepting
                            if not np.any(np.isinf(pcov_segment)) and not np.any(np.isnan(pcov_segment)) and \
                                    pcov_segment[0, 0] > 0:
                                best_r_squared = r_squared
                                best_segment_indices = (i, j)
                                best_popt = popt_segment
                                best_pcov = pcov_segment


                    except (RuntimeError, ValueError):
                        # Fit failed for this segment, ignore it
                        continue

            # --- Final Result ---
            if best_popt is None:
                print("Warning: Could not find any suitable linear segment for regression.")
                # Optionally, try fitting the whole range as a fallback? Or just fail.
                return np.nan

            fractal_dimension = best_popt[0]
            intercept = best_popt[1]  # Keep if needed

            print(f"Best linear segment found: Indices {best_segment_indices[0]} to {best_segment_indices[1] - 1}")
            print(
                f"  Scales from ~{valid_scales_arr[best_segment_indices[0]]:.2e} to ~{valid_scales_arr[best_segment_indices[1] - 1]:.2e}")
            print(f"  R-squared for this segment: {best_r_squared:.4f}")
            if best_r_squared < 0.95:  # Give a warning if the best fit isn't great
                print(f"Warning: Best R-squared ({best_r_squared:.4f}) is relatively low. Result may be less reliable.")

            print(f"Estimated Fractal Dimension (from best segment): {float(fractal_dimension)}")  # Final print

            return float(fractal_dimension)


        except MemoryError:
            print("MemoryError: Dataset likely too large or dimension too high for available memory.")
            return np.nan
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()  # Print detailed traceback for debugging
            return np.nan

    def compute_nd_fractal_dimension_batched(self, data_points: np.ndarray, max_points_per_batch: int = 10000) -> float:
        """
        Batch-processing version of the n-dimensional fractal dimension method for extremely large datasets.

        :param data_points: n-dimensional numpy array with format [x1, x2, x3, ..., label]
        :param max_points_per_batch: Maximum number of points to process in a single batch
        :return: Estimated fractal dimension
        """

        if len(data_points) == 0:
            return -1.0

        # If dataset is small enough, just use the regular method
        if len(data_points) <= max_points_per_batch:
            return self.compute_nd_fractal_dimension(data_points)

        try:
            # Process the data in batches
            num_batches = (len(data_points) + max_points_per_batch - 1) // max_points_per_batch
            print(f"Processing large dataset in {num_batches} batches...")

            # Take a uniform random sample across all batches to maintain distribution
            indices = np.random.choice(len(data_points), max_points_per_batch * min(num_batches, 5), replace=False)
            sampled_data = data_points[indices]

            # Use the regular method on the sampled data
            return self.compute_nd_fractal_dimension(sampled_data)

        except Exception as e:
            print(f"Error in batch processing: {e}")
            return -1.0

    def compute_hausdorff_dimension_enhanced(self, data_points: np.ndarray,
                                             max_sample_size: int = 5000,
                                             min_radius_factor: float = 0.001,
                                             max_radius_factor: float = 0.5,
                                             num_radii: int = 50,
                                             num_trials: int = 5,
                                             robust: bool = True,
                                             debug: bool = False) -> float:
        """
        High-precision estimator for the Hausdorff dimension using multiple enhanced techniques.

        :param data_points: N-dimensional numpy array with format [x1, x2, x3, ..., label]
        :param max_sample_size: Maximum number of points to use per trial
        :param min_radius_factor: Minimum radius as a fraction of data diameter
        :param max_radius_factor: Maximum radius as a fraction of data diameter
        :param num_radii: Number of radius values to use in estimation
        :param num_trials: Number of independent trials to average (reduces variance)
        :param robust: Whether to use robust regression methods
        :param debug: Whether to print debug information
        :return: Enhanced estimate of Hausdorff dimension
        """
        try:
            # Ensure data is 2D array
            if data_points.ndim == 3 and data_points.shape[0] == 1:
                data_points = data_points[0]

            # Extract coordinates (all columns except the last one which is the label)
            if data_points.shape[1] > 1:  # Make sure there are at least 2 columns
                coordinates = data_points[:, :-1].astype(np.float64)  # Use double precision for accuracy
            else:
                coordinates = data_points.astype(np.float64)

            if debug:
                print(f"Computing enhanced Hausdorff dimension for {coordinates.shape} data...")
                print(f"Using {num_trials} trials with {num_radii} radius values each")

            # Check if we have enough points
            if len(coordinates) < 50:
                print("Warning: Very few points for Hausdorff dimension. Results may be unreliable.")

            # Run multiple trials and average results (reduces sampling variance)
            dimensions = []
            r_squared_values = []

            for trial in range(num_trials):
                if debug and num_trials > 1:
                    print(f"Trial {trial + 1}/{num_trials}")

                # Sample dataset for this trial
                if len(coordinates) > max_sample_size:
                    indices = np.random.choice(len(coordinates), max_sample_size, replace=False)
                    trial_coords = coordinates[indices]
                else:
                    trial_coords = coordinates

                # Compute pairwise distances for this trial
                from scipy.spatial.distance import pdist

                # Try to use full distance matrix for smaller datasets
                if len(trial_coords) <= 2000:
                    distances = pdist(trial_coords)
                    max_distance = np.max(distances)

                    # Use adaptive radius selection based on distribution of distances
                    # This helps ensure we're capturing the scaling behavior properly
                    dist_percentiles = np.percentile(distances, [
                        min_radius_factor * 100,
                        max_radius_factor * 100
                    ])

                    min_radius = max(dist_percentiles[0], max_distance * 0.001)
                    max_radius = min(dist_percentiles[1], max_distance * 0.9)

                    # Generate more radii points at smaller scales where most interesting behavior happens
                    radii = np.concatenate([
                        np.logspace(np.log10(min_radius), np.log10(min_radius * 10), num=num_radii // 3),
                        np.logspace(np.log10(min_radius * 10), np.log10(max_radius), num_radii - num_radii // 3)
                    ])
                    radii = np.unique(radii)  # Remove duplicates

                    # Calculate correlation sum C(r) for each radius
                    C_r = []
                    for r in radii:
                        # Count pairs with distance < r
                        count = np.sum(distances < r)
                        # Normalize by total possible pairs
                        C_r.append(count / (len(trial_coords) * (len(trial_coords) - 1)))
                else:
                    # For larger datasets, use k-nearest neighbors approach
                    from sklearn.neighbors import NearestNeighbors

                    # Sample to estimate data diameter
                    sample_size = min(2000, len(trial_coords))
                    sample_idx = np.random.choice(len(trial_coords), sample_size, replace=False)
                    sample_dist = pdist(trial_coords[sample_idx])
                    max_distance = np.max(sample_dist)

                    # Use adaptive radius selection
                    dist_percentiles = np.percentile(sample_dist, [
                        min_radius_factor * 100,
                        max_radius_factor * 100
                    ])

                    min_radius = max(dist_percentiles[0], max_distance * 0.001)
                    max_radius = min(dist_percentiles[1], max_distance * 0.9)

                    # Generate more radii at smaller scales
                    radii = np.concatenate([
                        np.logspace(np.log10(min_radius), np.log10(min_radius * 10), num=num_radii // 3),
                        np.logspace(np.log10(min_radius * 10), np.log10(max_radius), num_radii - num_radii // 3)
                    ])
                    radii = np.unique(radii)

                    # Use efficient radius neighbor counting
                    nbrs = NearestNeighbors(algorithm='ball_tree').fit(trial_coords)

                    C_r = []
                    for r in radii:
                        # Find number of neighbors within radius r
                        count = 0
                        # Process in smaller batches to save memory
                        batch_size = 500
                        for i in range(0, len(trial_coords), batch_size):
                            batch_end = min(i + batch_size, len(trial_coords))
                            batch = trial_coords[i:batch_end]
                            dists, _ = nbrs.radius_neighbors(batch, radius=r, return_distance=True)
                            # Sum neighbors (excluding self)
                            batch_count = sum(len(d) - 1 for d in dists)  # -1 to exclude self
                            count += batch_count

                        # Normalize by total possible pairs
                        C_r.append(count / (len(trial_coords) * (len(trial_coords) - 1)))

                # Convert to logarithmic scale
                log_C_r = np.log(C_r)
                log_radii = np.log(radii)

                # Filter out invalid values
                valid_indices = np.isfinite(log_C_r)
                log_C_r = log_C_r[valid_indices]
                log_radii = log_radii[valid_indices]

                if len(log_C_r) < 5:
                    print("Warning: Too few valid points for regression in this trial")
                    continue

                # Fit regression line with adaptive selection of the scaling region
                # This is a key improvement - we automatically find the most linear region

                # Start with full range, then try to find best linear subrange
                best_r_squared = 0
                best_dim = 0

                # Try different subranges to find most linear scaling region
                min_points = max(5, len(log_radii) // 4)  # At least 5 points or 1/4 of all points

                for start_idx in range(len(log_radii) - min_points + 1):
                    for end_idx in range(start_idx + min_points, len(log_radii) + 1):
                        sub_log_radii = log_radii[start_idx:end_idx]
                        sub_log_C_r = log_C_r[start_idx:end_idx]

                        # Apply weights that emphasize the middle of the range
                        weights = np.ones_like(sub_log_radii)
                        if len(weights) > 4:
                            # Bell-shaped weighting gives more weight to the middle
                            rel_positions = np.linspace(0, 1, len(weights))
                            # Create bell curve weights
                            weights = np.exp(-4 * (rel_positions - 0.5) ** 2)

                        if robust:
                            # Use RANSAC for robust regression (resistant to outliers)
                            from sklearn.linear_model import RANSACRegressor

                            ransac = RANSACRegressor()
                            ransac.fit(sub_log_radii.reshape(-1, 1), sub_log_C_r, sample_weight=weights)
                            slope = ransac.estimator_.coef_[0]

                            # Calculate R-squared
                            y_pred = ransac.predict(sub_log_radii.reshape(-1, 1))
                            r_squared = 1 - (np.sum((sub_log_C_r - y_pred) ** 2) /
                                             np.sum((sub_log_C_r - np.mean(sub_log_C_r)) ** 2))
                        else:
                            # Use weighted polyfit
                            coeffs = np.polyfit(sub_log_radii, sub_log_C_r, 1, w=weights)
                            slope = coeffs[0]

                            # Calculate R-squared
                            p = np.poly1d(coeffs)
                            y_pred = p(sub_log_radii)
                            r_squared = 1 - (np.sum((sub_log_C_r - y_pred) ** 2) /
                                             np.sum((sub_log_C_r - np.mean(sub_log_C_r)) ** 2))

                        # Keep track of best fit
                        if r_squared > best_r_squared and len(sub_log_radii) >= min_points:
                            best_r_squared = r_squared
                            best_dim = slope

                            if debug and r_squared > 0.995:
                                # If we found an excellent fit, we can stop searching
                                break

                # Store results from this trial
                dimensions.append(best_dim)
                r_squared_values.append(best_r_squared)

                if debug:
                    print(f"Trial {trial + 1} dimension: {best_dim:.4f} (R² = {best_r_squared:.4f})")

            # Aggregate results across trials
            if not dimensions:
                print("Error: No valid dimension estimates obtained")
                return -1.0

            # Use weighted average based on R-squared values
            weights = np.array(r_squared_values) ** 2  # Square R² to emphasize better fits
            hausdorff_dimension = np.average(dimensions, weights=weights)

            # Calculate confidence interval
            from scipy import stats
            ci_95 = stats.t.interval(0.95, len(dimensions) - 1,
                                     loc=np.mean(dimensions),
                                     scale=stats.sem(dimensions))

            if debug:
                print("\nResults from all trials:")
                for i, (dim, r2) in enumerate(zip(dimensions, r_squared_values)):
                    print(f"  Trial {i + 1}: D = {dim:.4f} (R² = {r2:.4f})")
                print(f"\nFinal Hausdorff dimension estimate: {hausdorff_dimension:.4f}")
                print(f"95% Confidence interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
                print(f"Standard deviation: {np.std(dimensions):.4f}")

            return hausdorff_dimension

        except Exception as e:
            print(f"Error in enhanced Hausdorff calculation: {e}")
            import traceback
            traceback.print_exc()
            return -1.0

    @staticmethod
    def get_available_cores(threshold: int = 50) -> int:
        """
        Get the number of available CPU cores based on current CPU usage.

        Args:
            threshold (int): The CPU usage percentage threshold to consider a core as available.

        Returns:
            int: The number of available cores that are not heavily used.
        """
        num_cores = multiprocessing.cpu_count()
        current_usage = psutil.cpu_percent(interval=1)

        # If current CPU usage is above the threshold, reduce the number of available cores
        available_cores = num_cores
        if current_usage > threshold:
            # Reduce the number of available cores based on the usage
            available_cores = max(1, num_cores - int(current_usage / 100 * num_cores))

        return available_cores

    def detect(self, create_boundary_type_only: bool = False) -> None:
        """
        Computes the fractal dimension of each boundary with each boundary type being assigned a separate core for processing using multiprocessing.

        Requires:
        -----

        'Neighbour_Boundary.npy' and 'BoundaryExtractor.py' to be there in the directory.

        :param create_boundary_type_only: takes a boolean value in case we want to only create the 'Boundary_Type.npy' file and not both.
        :return: None
        """

        # Get the number of available CPU cores based on current usage
        num_workers = FractalDetector.get_available_cores(threshold=50)

        boundaries = []
        boundary_type = []

        for i in range(1, self.__num_structs):

            for j in range(i + 1, self.__num_structs + 1):

                boundary = BoundaryExtractor.fetch(mini_grid_num=-1, lower_struct=i, higher_struct=j, filter_boundary=True)
                if len(boundary) == 0:
                    boundaries.append(np.asarray([]))
                    boundary_type.append([i, j])
                    continue
                joined_boundary = np.vstack(boundary)
                joined_boundary = np.array(list({tuple(point) for point in joined_boundary}), dtype=float)
                joined_boundary = joined_boundary[:, :-1]

                boundaries.append(joined_boundary)
                boundary_type.append([i, j])


        boundary_type = np.asarray(boundary_type)

        params = list(zip(boundaries))

        if not create_boundary_type_only:

            # Create a Pool with the calculated number of available CPU cores
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Use pool.starmap to unpack the parameter tuples
                if self.__num >= 0:
                    dimensions = pool.starmap(self.compute_nd_fractal_dimension, params)
                else:
                    dimensions = pool.starmap(self.compute_hausdorff_dimension_enhanced, params)

            dimensions = np.asarray(dimensions)

            np.save("Dimensions", dimensions)

        np.save("Boundary_Type.npy", boundary_type)

    def detect_general(self, points) -> float:

        if self.__num >= 0:
            return self.compute_nd_fractal_dimension(points[:, :-1])
        else:
            return self.compute_hausdorff_dimension_enhanced(points[:, :-1])