import matplotlib.pyplot as plt
import numpy as np

from qts2d.encoding.qgaf import QGAF

# Generate a sample time series
n_timestamps = 100
ts = np.sin(np.linspace(0, 10 * np.pi, n_timestamps)) + np.random.normal(0, 0.2, n_timestamps)
X = ts.reshape(1, -1)  # Reshape to (n_samples, n_timestamps)

# QGAF Transformers
# Example 1: QGASF with Hadamard computation
qgasf_hadamard = QGAF(image_size=32, method='summation', computation_method='hadamard', scaling='minmax')
X_qgasf_hadamard = qgasf_hadamard.fit_transform(X)

# Example 2: QGADF with Hadamard computation
qgadf_hadamard = QGAF(image_size=32, method='difference', computation_method='hadamard', scaling='minmax')
X_qgadf_hadamard = qgadf_hadamard.fit_transform(X)

# Example 3: QGASF with Xu computation (no scaling)
qgasf_xu_none = QGAF(image_size=32, method='summation', computation_method='xu', scaling=None)
X_qgasf_xu_none = qgasf_xu_none.fit_transform(X)

# Example 4: QGADF with Xu computation (standard scaling)
qgadf_xu_std = QGAF(image_size=32, method='difference', computation_method='xu', scaling='standard')
X_qgadf_xu_std = qgadf_xu_std.fit_transform(X)


# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 11))

# Plot original time series
axs[0, 0].plot(ts)
axs[0, 0].set_title("Original Time Series")
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Value")

# Plot QGASF (Hadamard)
img1 = axs[0, 1].imshow(X_qgasf_hadamard[0], cmap='coolwarm', origin='lower')
axs[0, 1].set_title("QGASF (Hadamard, minmax)")
fig.colorbar(img1, ax=axs[0, 1], fraction=0.046, pad=0.04)

# Plot QGADF (Hadamard)
img2 = axs[0, 2].imshow(X_qgadf_hadamard[0], cmap='coolwarm', origin='lower')
axs[0, 2].set_title("QGADF (Hadamard, minmax)")
fig.colorbar(img2, ax=axs[0, 2], fraction=0.046, pad=0.04)

# Plot QGASF (Xu, None)
img3 = axs[1, 0].imshow(X_qgasf_xu_none[0], cmap='viridis', origin='lower')
axs[1, 0].set_title("QGASF (Xu, no scaling)")
fig.colorbar(img3, ax=axs[1, 0], fraction=0.046, pad=0.04)

# Plot QGADF (Xu, Standard)
img4 = axs[1, 1].imshow(X_qgadf_xu_std[0], cmap='viridis', origin='lower')
axs[1, 1].set_title("QGADF (Xu, standard scaling)")
fig.colorbar(img4, ax=axs[1, 1], fraction=0.046, pad=0.04)

# Remove empty subplot
fig.delaxes(axs[1,2])

plt.tight_layout()
plt.show()

print(f"Original time series shape: {X.shape}")
print(f"QGASF (Hadamard) output shape: {X_qgasf_hadamard.shape}")
print(f"QGADF (Hadamard) output shape: {X_qgadf_hadamard.shape}")
print(f"QGASF (Xu, None) output shape: {X_qgasf_xu_none.shape}")
print(f"QGADF (Xu, Standard) output shape: {X_qgadf_xu_std.shape}")