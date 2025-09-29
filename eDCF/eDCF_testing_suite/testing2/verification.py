import numpy as np

data = np.load("manifold_reference_model.npy", allow_pickle=True).item()

for key, value in sorted(data.items()):
    temp = 3 ** key - 1
    print(f"{key}: {value} - {temp}")