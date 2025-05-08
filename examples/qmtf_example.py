import matplotlib.pyplot as plt
import numpy as np

from qts2d.encoding import QMTF

# 1. Generate Sample Data
n_samples = 1
n_timestamps = 100
X = np.random.rand(n_samples, n_timestamps)

# Optional: A more structured signal for better visualization
# X[0, :50] = np.sin(np.linspace(0, 4 * np.pi, 50)) * 0.5 + 0.5
# X[0, 50:] = np.cos(np.linspace(0, 2 * np.pi, 50)) * 0.3 + 0.2

# 2. Instantiate Transformer
# Using image_size as an int for a fixed plot size, and flatten=False
qmtf = QMTF(image_size=32, n_bins=8, flatten=False, scaling='minmax')

# 3. Fit and Transform
X_qmtf = qmtf.fit_transform(X)

# 4. Plot the result
# X_qmtf shape is (n_samples, image_size, image_size)
image_to_plot = X_qmtf[0]

plt.figure(figsize=(6, 6))
plt.imshow(image_to_plot, cmap='viridis', origin='lower')
plt.title(f'QMTF Output (Image Size: {qmtf._image_size_internal_}, Bins: {qmtf.n_bins})')
plt.xlabel("Bin Index")
plt.ylabel("Bin Index")
plt.colorbar(label="Transition Probability")
plt.tight_layout()
# plt.savefig("plot_qmtf_example.png")
plt.show()

print(f"QMTF Input shape: {X.shape}")
print(f"QMTF Output shape: {X_qmtf.shape}")