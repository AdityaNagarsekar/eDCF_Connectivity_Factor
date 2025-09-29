import numpy as np
import skdim.id as id
from skdim.datasets import BenchmarkManifolds
import warnings

# --- Suppress potential warnings from skdim for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Generate All Benchmark Manifolds ---
print("Generating benchmark manifolds...")
bm = BenchmarkManifolds(random_state=42)
# Using slightly more points for better estimates
all_manifolds = bm.generate(name='all', n=15000, noise=0.01) 
print(f"Generated {len(all_manifolds)} manifolds.")

# --- 2. Process each manifold with skdim estimators ---
print("\nEstimating intrinsic dimension using skdim...")
results = []

# Instantiate the estimators once outside the loop
mle_estimator = id.MLE()
twonn_estimator = id.TwoNN()

for name, data in all_manifolds.items():
    print(f"--- Processing: {name} ---")
    
    if data is None or data.shape[0] < 20: # Ensure enough points for k=20
        print(f"Skipping {name} due to insufficient data.")
        continue

    # Get the ground truth intrinsic dimension from the benchmark object
    known_id = bm.truth.loc[name, "Intrinsic Dimension"]
    
    # --- Estimate dimension using skdim.id.MLE ---
    try:
        # CORRECTED: Use fit_predict and the 'n_neighbors' argument
        mle_dim = mle_estimator.fit_predict(data, n_neighbors=20)
        # Round the result for clearer comparison
        mle_dim_rounded = int(np.round(mle_dim))
    except Exception as e:
        print(f"MLE failed for {name}: {e}")
        mle_dim_rounded = "Error"

    # --- Estimate dimension using skdim.id.TwoNN ---
    try:
        # TwoNN does not require a neighbor parameter
        twonn_dim = twonn_estimator.fit_transform(data)
        # Round the result for clearer comparison
        twonn_dim_rounded = int(np.round(twonn_dim))
    except Exception as e:
        print(f"TwoNN failed for {name}: {e}")
        twonn_dim_rounded = "Error"

    results.append([name, known_id, mle_dim_rounded, twonn_dim_rounded])


# --- 3. Display Results ---
print("\n\n--- skdim Verification Results ---")
print(f"{'Manifold':<20} | {'Known Intrinsic Dim':<20} | {'skdim MLE Estimate':<20} | {'skdim TwoNN Estimate'}")
print("-" * 85)
for res in results:
    print(f"{res[0]:<20} | {res[1]:<20} | {str(res[2]):<20} | {str(res[3])}")

print("\nVerification complete.")