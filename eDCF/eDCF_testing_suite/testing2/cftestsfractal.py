import numpy as np
import os
import warnings
import skdim.id as id
from tqdm import tqdm
import random
from typing import Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

# --- Import custom classes ---
# Ensure these files (ForceGridObject.py, Connectivity.py, Weight.py) are in the same directory
from ForceGridObject import ForceGridObject
from Connectivity import Connectivity
from Weight import Weight

# ==============================================================================
# 1. JULIA SET CLASS (Provided by you)
# ==============================================================================
class JuliaSet:
    """
    Generates Julia Set boundary points for visualization.
    """

    def __init__(
            self,
            identity: int,
            iterations: int = 100000,
            boundary_threshold: float = 0.01,
            c: complex = -0.122561 + 0.744861j,
            max_iter: int = 100,
            escape_radius: float = 2.0
    ):
        self.__identity = identity
        self.__iterations = iterations
        self.__boundary_threshold = boundary_threshold
        self.__c = c
        self.__max_iter = max_iter
        self.__escape_radius = escape_radius
        self.__datapoints = np.empty((0, 3), float)

    def generate(self) -> np.ndarray:
        x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0
        boundary_points = []
        grid_size = int(np.sqrt(self.__iterations * 10))
        x_vals = np.linspace(x_min, x_max, grid_size)
        y_vals = np.linspace(y_min, y_max, grid_size)
        count = 0
        for x in x_vals:
            for y in y_vals:
                if count >= self.__iterations: break
                z = complex(x, y)
                if self.is_boundary_point(z):
                    boundary_points.append((x, y))
                    count += 1
            if count >= self.__iterations: break
        while len(boundary_points) < self.__iterations:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = complex(x, y)
            if self.is_boundary_point(z):
                boundary_points.append((x, y))
        boundary_array = np.array(boundary_points[:self.__iterations])
        labels = np.full((boundary_array.shape[0], 1), self.__identity)
        self.__datapoints = np.hstack((boundary_array, labels))
        return self.__datapoints
    
    def is_boundary_point(self, z: complex) -> bool:
        center_escapes = self.escape_time(z) < self.__max_iter
        delta = self.__boundary_threshold
        neighbors = [z + delta, z - delta, z + delta * 1j, z - delta * 1j]
        for neighbor in neighbors:
            if (self.escape_time(neighbor) < self.__max_iter) != center_escapes:
                return True
        return False

    def escape_time(self, z: complex) -> int:
        c = self.__c
        n = 0
        while abs(z) < self.__escape_radius and n < self.__max_iter:
            z = z * z + c
            n += 1
        return n

# ==============================================================================
# 2. NEW SHAPE GENERATOR FUNCTIONS
# ==============================================================================

def generate_circle(n_samples: int, noise: float = 0.01) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n_samples)
    x = np.cos(angles)
    y = np.sin(angles)
    data = np.vstack((x, y)).T
    data += np.random.normal(scale=noise, size=data.shape)
    return data

def generate_barnsley_fern(n_samples: int) -> np.ndarray:
    points = np.zeros((n_samples, 2))
    x, y = 0, 0
    for i in range(1, n_samples):
        r = random.random()
        if r < 0.01: x, y = 0, 0.16 * y
        elif r < 0.86: x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6
        elif r < 0.93: x, y = 0.2 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6
        else: x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44
        points[i] = [x, y]
    points /= 10.0
    return points

def generate_sierpinski_carpet(n_samples: int) -> np.ndarray:
    points = np.zeros((n_samples, 2))
    x, y = random.random(), random.random()
    transformations = [
        lambda x_val, y_val: (x_val / 3, y_val / 3), lambda x_val, y_val: (x_val / 3 + 1/3, y_val / 3),
        lambda x_val, y_val: (x_val / 3 + 2/3, y_val / 3), lambda x_val, y_val: (x_val / 3, y_val / 3 + 1/3),
        lambda x_val, y_val: (x_val / 3 + 2/3, y_val / 3 + 1/3), lambda x_val, y_val: (x_val / 3, y_val / 3 + 2/3),
        lambda x_val, y_val: (x_val / 3 + 1/3, y_val / 3 + 2/3), lambda x_val, y_val: (x_val / 3 + 2/3, y_val / 3 + 2/3),
    ]
    for i in range(n_samples):
        x, y = random.choice(transformations)(x, y)
        points[i] = [x, y]
    return points

def generate_julia(n_samples: int) -> np.ndarray:
    julia_gen = JuliaSet(identity=1, iterations=n_samples)
    data_with_label = julia_gen.generate()
    return data_with_label[:, :2]

# ==============================================================================
# 3. YOUR CUSTOM PIPELINE FUNCTION (MODIFIED TO RETURN WEIGHTS)
# ==============================================================================

def run_custom_pipeline(data: np.ndarray, target_percentage: float) -> Tuple[Union[str, int], Optional[Dict[int, float]]]:
    """
    Runs the custom pipeline and returns both the estimated dimension and the normalized weights.
    """
    try:
        data_normalized = data.copy()
        min_bounds = np.min(data_normalized, axis=0)
        max_bounds = np.max(data_normalized, axis=0)
        ranges = max_bounds - min_bounds
        ranges[ranges == 0] = 1
        global_range = np.max(ranges)
        if global_range == 0: global_range = 1
        data_normalized = (data_normalized - min_bounds) / global_range
        
        n_samples, ambient_dim = data_normalized.shape
        adaptive_target = min(95.0, target_percentage + np.sqrt(ambient_dim) * 3)

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
            if current_percentage < adaptive_target: high_sp = mid_sp
            else: low_sp = mid_sp
        best_spacing = (low_sp + high_sp) / 2
        os.remove(temp_data_file)

        final_grid_info = np.array([best_spacing, 0.1, 2, ambient_dim])
        grid_set = fgo.direct_gridder(data_normalized, grid_bounds, final_grid_info, best_spacing)
        grid_points = np.array(list(grid_set))

        if grid_points.shape[0] < 2:
            return "N/A (Grid)", None

        _, weight_obj = Connectivity.calc_connectivity_general([grid_points], spacing=best_spacing)

        if not weight_obj or not weight_obj.weights:
            return "Error (Weight)", None

        weights_dict = weight_obj.weights
        dimensions = np.array(list(weights_dict.keys()))
        weights = np.array(list(weights_dict.values()))

        total_weight = np.sum(weights)
        if total_weight > 0:
            normalized_weights = weights / total_weight
            weighted_avg_dim = np.sum(dimensions * normalized_weights)
            # Create a dictionary of the normalized weights to return
            normalized_weights_dict = dict(zip(dimensions, normalized_weights))
            return int(np.round(weighted_avg_dim)), normalized_weights_dict
        else:
            return 0, None

    except Exception as e:
        print(f"Caught exception: {e}")
        if os.path.exists("Datapoints.npy"): os.remove("Datapoints.npy")
        return "Error", None

# ==============================================================================
# 4. PLOTTING FUNCTION
# ==============================================================================

def plot_shape(data: np.ndarray, title: str):
    """Plots a 2D dataset and saves it as a high-quality PNG."""
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], s=0.01, alpha=0.7, c='blue')
    plt.title(title, fontsize=16)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure with high resolution and a tight bounding box
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close() # Close the plot to free up memory


# ==============================================================================
# 5. MAIN EXECUTION BLOCK (Modified for printing weights)
# ==============================================================================

if __name__ == '__main__':
    num_points = 100000

    print(f"Generating {num_points} points for each custom shape...")
    all_shapes = {
        'Circle': generate_circle(n_samples=num_points, noise=0.001),
        'Barnsley Fern': generate_barnsley_fern(n_samples=num_points),
        'Julia Set': generate_julia(n_samples=num_points),
        'Sierpinski Carpet': generate_sierpinski_carpet(n_samples=num_points)
    }
    known_ids = {
        'Circle': 1,
        'Barnsley Fern': 1,
        'Julia Set': 1,
        'Sierpinski Carpet': 1
    }
    print(f"Generated {len(all_shapes)} custom shapes.")

    print("\nPlotting generated shapes (saving to files)...")
    for name, data in all_shapes.items():
        plot_shape(data, name)

    mle_estimator = id.MLE()
    twonn_estimator = id.TwoNN()
    
    target_percentages = [50]
    for target_perc in target_percentages:
        print(f"\n\n--- Running Benchmark for Target Grid Density: ~{target_perc}% ---")
        results = []
        
        for name, data in tqdm(all_shapes.items(), desc=f"Target {target_perc}%"):
            if data is None or data.shape[0] < 20: continue
            known_id = known_ids[name]
            
            try: mle_dim = int(np.round(mle_estimator.fit_predict(data, n_neighbors=20)))
            except Exception: mle_dim = "Error"
            try: twonn_dim = int(np.round(twonn_estimator.fit_transform(data)))
            except Exception: twonn_dim = "Error"

            # Unpack the result and the weights
            custom_td_result, norm_weights = run_custom_pipeline(data, target_percentage=target_perc)
            results.append([name, known_id, mle_dim, twonn_dim, custom_td_result, norm_weights])

        print(f"\n--- Results for Target {target_perc}% ---")
        print(f"{'Shape':<20} | {'Known ID':<10} | {'skdim MLE':<10} | {'skdim TwoNN':<12} | {'Custom TD (Fine-Tuned)'}")
        print("-" * 105)
        for res in results:
            # res[0]=name, res[1]=known_id, ..., res[4]=result, res[5]=weights
            print(f"{res[0]:<20} | {str(res[1]):<10} | {str(res[2]):<10} | {str(res[3]):<12} | {str(res[4])}")
            
            # Print the normalized weights if they exist
            if res[5] is not None:
                # Format the dictionary for clean printing
                weights_str = ", ".join([f"dim {d}: {w:.2%}" for d, w in sorted(res[5].items())])
                print(f"  └─ Normalized Weights: {weights_str}")
        print("-" * 105)


    print("\n\nVerification complete.")