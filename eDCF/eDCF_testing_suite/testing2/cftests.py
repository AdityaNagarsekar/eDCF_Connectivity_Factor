import numpy as np
import os
import warnings
import skdim.id as id
from skdim.datasets import BenchmarkManifolds
from tqdm import tqdm

# --- Import custom classes and the new reference generator ---
from ForceGridObject import ForceGridObject
from Connectivity import Connectivity
from Weight import Weight
import generate_reference_model

def run_custom_pipeline(data: np.ndarray, target_percentage: float):
    """
    Runs the custom pipeline with data normalization and adaptive spacing,
    using a weighted average of the final weight distribution for dimension estimation.
    """
    try:
        # --- Step 1: Normalize Data ---
        data_normalized = data.copy()
        min_bounds = np.min(data_normalized, axis=0)
        max_bounds = np.max(data_normalized, axis=0)
        ranges = max_bounds - min_bounds
        ranges[ranges == 0] = 1
        global_range = np.max(ranges)
        if global_range == 0: global_range = 1
        data_normalized = (data_normalized - min_bounds) / global_range
        
        n_samples, ambient_dim = data_normalized.shape
        
        # --- Step 2: MODIFIED - Damped Adaptive Grid Target ---
        # This heuristic is less aggressive for high-dimensional data, promoting stability.
        adaptive_target = min(95.0, target_percentage + np.sqrt(ambient_dim) * 3)

        # sample size normalised scaling:
        # scaling_factor = np.sqrt(ambient_dim) * (1 / np.log1p(n_samples))
        # adaptive_target = min(85.0, target_percentage + scaling_factor * 3)

        #dimension aware:
        # scaling = (np.sqrt(ambient_dim) / np.log1p(ambient_dim)) * 2.5
        # adaptive_target = min(75.0, target_percentage + scaling)

        # None:
        # adaptive_target = target_percentage

        # --- Step 3: Adaptive Spacing ---
        temp_data_file = "Datapoints.npy"
        np.save(temp_data_file, [np.hstack([data_normalized, np.ones((n_samples, 1))])])
        fgo = ForceGridObject(identity=1)
        grid_bounds = np.vstack((np.min(data_normalized, axis=0), np.max(data_normalized, axis=0))).T
        
        low_sp, high_sp = 0.001, 0.5
        for _ in range(15):
            mid_sp = (low_sp + high_sp) / 2
            if mid_sp < 1e-6: break
            grid_info = np.array([mid_sp, 0.1, 2, ambient_dim])
            grid_set = fgo.direct_gridder(data_normalized, grid_bounds, grid_info, mid_sp)
            current_percentage = len(grid_set) / n_samples * 100
            if current_percentage < adaptive_target:
                high_sp = mid_sp
            else:
                low_sp = mid_sp
        best_spacing = (low_sp + high_sp) / 2
        os.remove(temp_data_file)

        # --- Step 4: Gridding and Connectivity ---
        final_grid_info = np.array([best_spacing, 0.1, 2, ambient_dim])
        grid_set = fgo.direct_gridder(data_normalized, grid_bounds, final_grid_info, best_spacing)
        grid_points = np.array(list(grid_set))

        if grid_points.shape[0] < 2:
            return "N/A (Grid)"

        _, weight_obj = Connectivity.calc_connectivity_general([grid_points], spacing=best_spacing)

        # --- Step 5: Dimension Estimation using Weighted Average ---
        if not weight_obj or not weight_obj.weights:
            return "Error (Weight)"

        weights_dict = weight_obj.weights
        dimensions = np.array(list(weights_dict.keys()))
        weights = np.array(list(weights_dict.values()))

        total_weight = np.sum(weights)
        if total_weight > 0:
            normalized_weights = weights / total_weight
            # Calculate the weighted average of dimensions
            weighted_avg_dim = np.sum(dimensions * normalized_weights)
            return int(np.round(weighted_avg_dim))
        else:
            return 0

    except Exception as e:
        print(f"Caught exception: {e}") # Added for better debugging
        if os.path.exists("Datapoints.npy"): os.remove("Datapoints.npy")
        return "Error"

if __name__ == '__main__':
    for i in [1000]:
        # --- Step 1 - Generate the reference model first ---
        # IMPORTANT: Delete the old 'manifold_reference_model.npy' to trigger regeneration.
        # if not os.path.exists('manifold_reference_model.npy'):
        #     generate_reference_model.main(i)
        # else:
        #     # print("Using existing manifold reference model.")
        #     generate_reference_model.main(i)
        generate_reference_model.main(i)
            
        # --- Step 2: Main Benchmark Execution ---
        print("\nGenerating benchmark manifolds...")
        bm = BenchmarkManifolds(random_state=42)
        all_manifolds = bm.generate(name='all', n=i, noise=0.1)
        print(f"Generated {len(all_manifolds)} manifolds.")

        mle_estimator = id.MLE()
        twonn_estimator = id.TwoNN()
        
        target_percentages = [50]
        for target_perc in target_percentages:
            print(f"\n\n--- Running Benchmark for Target Grid Density: ~{target_perc}% ---")
            results = []
            
            for name, data in tqdm(all_manifolds.items(), desc=f"Target {target_perc}%"):
                if data is None or data.shape[0] < 20: continue

                known_id = bm.truth.loc[name, "Intrinsic Dimension"]
                
                try: mle_dim = int(np.round(mle_estimator.fit_predict(data, n_neighbors=20)))
                except Exception: mle_dim = "Error"

                try: twonn_dim = int(np.round(twonn_estimator.fit_transform(data)))
                except Exception: twonn_dim = "Error"

                custom_td_result = run_custom_pipeline(data, target_percentage=target_perc)
                results.append([name, known_id, mle_dim, twonn_dim, custom_td_result])

            print(f"\n--- Results for Target {target_perc}% ---")
            print(f"{'Manifold':<20} | {'Known ID':<10} | {'skdim MLE':<10} | {'skdim TwoNN':<12} | {'Custom TD (Fine-Tuned)'}")
            print("-" * 95)
            for res in results:
                print(f"{res[0]:<20} | {str(res[1]):<10} | {str(res[2]):<10} | {str(res[3]):<12} | {str(res[4])}")

        print("\n\nVerification complete.")