import matplotlib.pyplot as plt
import numpy as np

from qts2d.encoding import QRP

# 1. Generate Sample Data
n_samples = 1
n_timestamps = 100 # Ensure n_timestamps is large enough for dimension and time_delay
X = np.random.rand(n_samples, n_timestamps)
# Optional: A signal with repeating patterns for better RP visualization
# s = np.sin(np.linspace(0, 4 * np.pi, 50))
# X[0, :] = np.concatenate((s, s[:25], s[10:35])) # Create some recurrences

# 2. Instantiate Transformer
# Using dimension=3, time_delay=2, threshold=None for continuous values, flatten=False
qrp = QRP(dimension=3, time_delay=2, threshold=None, percentage=10, flatten=False, scaling='minmax')

# 3. Fit and Transform
X_qrp = qrp.fit_transform(X)

# 4. Plot the result
# X_qrp shape is (n_samples, image_size, image_size)
image_to_plot = X_qrp[0]

plt.figure(figsize=(6, 6))
plt.imshow(image_to_plot, cmap='viridis', origin='lower')
plt.title(f'QRP Output (Dim: {qrp.dimension}, Delay: {qrp.time_delay}, Threshold: {qrp.threshold})')
plt.xlabel("Trajectory Index")
plt.ylabel("Trajectory Index")
plt.colorbar(label="Quantum Distance (sqrt(1-fidelity^2))" if qrp.threshold is None else "Recurrence (0 or 1)")
plt.tight_layout()
# plt.savefig("plot_qrp_example.png")
plt.show()

print(f"QRP Input shape: {X.shape}")
print(f"QRP Output shape: {X_qrp.shape}")
if X_qrp.ndim == 3 and X_qrp.shape[0] > 0:
    print(f"QRP effective image size: {X_qrp.shape[1]}x{X_qrp.shape[2]}")
