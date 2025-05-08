import matplotlib.pyplot as plt
import numpy as np

from qts2d.encoding import QSG

# 1. Generate Sample Data
n_samples = 1
n_timestamps = 1024 # Should be adequate for segment_length
X = np.random.rand(n_samples, n_timestamps)

# Optional: A signal with distinct frequencies for better spectrogram visualization
# t = np.linspace(0, 1, n_timestamps, endpoint=False)
# X[0, :] = np.sin(2 * np.pi * 10 * t) + 0.7 * np.sin(2 * np.pi * 25 * t) + 0.4 * np.sin(2 * np.pi * 50 * t)
# If using a custom signal, ensure it's compatible with the chosen scaling method,
# or adjust scaling. For 'minmax' as used in QSG default, values are scaled to [-1, 1].

# 2. Instantiate Transformer
# segment_length must be a power of 2. flatten=False for 2D plot.
qsg = QSG(segment_length=64, overlap_ratio=0.5, window_name='hann', shots=1024, flatten=False, scaling='minmax')

# 3. Fit and Transform
X_qsg = qsg.fit_transform(X)

# 4. Plot the result
# X_qsg shape is (n_samples, num_freq_bins, num_time_segments)
image_to_plot = X_qsg[0] # This is (num_freq_bins, num_time_segments)

plt.figure(figsize=(8, 6))
plt.imshow(image_to_plot, aspect='auto', cmap='viridis', origin='lower')
plt.title(f'QSG Output (Segment Length: {qsg.segment_length})')
plt.xlabel("Time Segment Index")
plt.ylabel("Frequency Bin Index")
plt.colorbar(label="Quantum Power Spectrum")
plt.tight_layout()
# plt.savefig("plot_qsg_example.png")
plt.show()

print(f"QSG Input shape: {X.shape}")
print(f"QSG Output shape: {X_qsg.shape}")
if X_qsg.ndim == 3 and X_qsg.shape[0] > 0:
    print(f"QSG: {X_qsg.shape[1]} Freq Bins, {X_qsg.shape[2]} Time Segments")
