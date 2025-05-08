"""Quantum Recurrence Plot (QRP) implementation."""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

import numpy as np
from numpy.lib.stride_tricks import as_strided
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.utils.validation import check_array, check_is_fitted

from ..base_transformer import BaseQuantumTransformer


# Adapted from pyts: https://github.com/johannfaouzi/pyts
def _trajectories(X, dimension, time_delay):
    """Extract trajectories from time series."""
    n_samples, n_timestamps = X.shape
    n_trajectories = n_timestamps - (dimension - 1) * time_delay
    if n_trajectories <= 0:
        return np.empty((n_samples, 0, dimension), dtype=X.dtype)

    shape_new = (n_samples, n_trajectories, dimension)
    s0, s1 = X.strides
    strides_new = (s0, s1, time_delay * s1)
    return as_strided(X, shape=shape_new, strides=strides_new)


class QRP(BaseQuantumTransformer):
    """Quantum Recurrence Plot transformer using Fidelity Kernel.

    Encodes time series into images representing quantum distances
    between trajectories extracted from the original time series. The distance
    is derived from the fidelity between quantum states encoded using a
    feature map (e.g., ZFeatureMap), calculated via FidelityStatevectorKernel.

    Parameters
    ----------
    dimension : int, default=1
        Dimension of the trajectory (embedding dimension). This also serves as
        the `feature_dimension` for the quantum feature map. Must be >= 1.
    time_delay : int, default=1
        Time gap between two back-to-back points of the trajectory. Must be >= 1.
    feature_map_reps : int, default=2
        Number of repetitions for the feature map circuit (e.g., `reps` for ZFeatureMap).
    threshold : float, 'point', 'distance' or None, default=None
        Threshold for the quantum distance to binarize the plot.
        Distance is calculated as `sqrt(1 - fidelity^2)`.
        - None: The recurrence plot contains the raw quantum distances.
        - 'point': Threshold computed such that `percentage`% of points are
          below it (closer than threshold).
        - 'distance': Threshold is `percentage`% of the maximum quantum distance
          found in the plot.
        - float: A fixed distance threshold. Points with distance <= threshold
          are marked as 1.
    percentage : float, default=10
        Percentage used when `threshold` is 'point' or 'distance'. Must be
        between 0 and 100.
    flatten : bool, default=False
        If True, flatten the output image into a 1D array.
    scaling : str or None, default='minmax'
        Time series scaling method applied *before* trajectory extraction and
        quantum encoding. Inherited from `BaseQuantumTransformer`. Options:
        'minmax', 'standard', None. Scaling is recommended for feature maps.

    References
    ----------
    .. [1] J.-P Eckmann, S. Oliffson Kamphorst and D Ruelle, "Recurrence
    Plots of Dynamical Systems". Europhysics Letters (1987).
    """
    def __init__(self, dimension=1, time_delay=1, feature_map_reps=2,
                 threshold=None, percentage=10, flatten=False, scaling='minmax'):
        super().__init__(scaling=scaling)
        self.dimension = dimension
        self.time_delay = time_delay
        self.feature_map_reps = feature_map_reps
        self.threshold = threshold
        self.percentage = percentage
        self.flatten = flatten

        if not isinstance(self.dimension, (int | np.integer)) or self.dimension < 1:
            raise ValueError("`dimension` must be a positive integer.")
        if not isinstance(self.time_delay, (int | np.integer)) or self.time_delay < 1:
            raise ValueError("`time_delay` must be a positive integer.")
        if not isinstance(self.feature_map_reps, (int | np.integer)) or self.feature_map_reps < 1:
            raise ValueError("`feature_map_reps` must be a positive integer.")

        if (self.threshold is not None
            and self.threshold not in ['point', 'distance']
            and not isinstance(self.threshold, (int | np.integer | float | np.floating))):
            raise TypeError("`threshold` must be None, 'point', 'distance', or a number.")
        if isinstance(self.threshold, (int | np.integer | float | np.floating)) and self.threshold < 0:
            raise ValueError("Numeric `threshold` must be non-negative.")

        if not isinstance(self.percentage, (int | np.integer | float | np.floating)) or not (0 <= self.percentage <= 100):
            raise ValueError("`percentage` must be between 0 and 100.")

        if not isinstance(self.flatten, bool):
             raise TypeError("`flatten` must be a boolean.")


    def _check_trajectory_params(self, n_timestamps):
        """Validate dimension and time_delay against timestamps and calculate sizes.

        Args:
            n_timestamps (int): Number of timestamps in the input time series.
        """
        self._n_trajectories = n_timestamps - (self.dimension - 1) * self.time_delay
        if self._n_trajectories < 1:
            raise ValueError(
                f"The number of trajectories ({self._n_trajectories}) must be positive. "
                f"Input time series length ({n_timestamps}) is too short for the "
                f"chosen `dimension` ({self.dimension}) and `time_delay` ({self.time_delay}). "
                f"Required length >= 1 + (dimension - 1) * time_delay = {1 + (self.dimension - 1) * self.time_delay}."
            )
        self._image_size = self._n_trajectories


    def _compute_qrp_matrix(self, ts):
        """Computes the Quantum Recurrence Plot matrix for a single time series
        using FidelityStatevectorKernel.

        Args:
            ts (np.ndarray): 1D array representing a time series segment.
        
        Returns:
            np.ndarray: The resulting n x n quantum recurrence plot matrix.
        """
        trajectories = _trajectories(ts.reshape(1, -1), self.dimension, self.time_delay)[0]

        if trajectories.shape[0] == 0:
             return np.zeros((0, 0))
        if trajectories.shape[0] == 1:
             return np.zeros((1, 1))

        feature_map = ZFeatureMap(feature_dimension=self.dimension, reps=self.feature_map_reps)

        kernel = FidelityStatevectorKernel(feature_map=feature_map)

        fidelity_matrix = kernel.evaluate(x_vec=trajectories)

        distance_matrix = np.sqrt(np.clip(1.0 - fidelity_matrix, 0.0, 1.0))

        np.fill_diagonal(distance_matrix, 0.0)

        if self.threshold is not None:
            if self.threshold == 'point':
                percentile_k = max(0.0, min(100.0, self.percentage))
                threshold_val = np.percentile(distance_matrix.ravel(), percentile_k)
                if np.isclose(threshold_val, 0) and percentile_k < 1:
                     threshold_val = np.finfo(distance_matrix.dtype).eps
            elif self.threshold == 'distance':
                max_dist = np.max(distance_matrix)
                if np.isclose(max_dist, 0):
                    threshold_val = 0
                else:
                    threshold_val = (self.percentage / 100.0) * max_dist
            else:
                threshold_val = float(self.threshold)

            binary_qrp_matrix = (distance_matrix <= threshold_val).astype(float)
            return binary_qrp_matrix
        else:
            return distance_matrix


    def fit(self, X, y=None):
        """Fit the QRP transformer.

        Validates input and parameters. Determines trajectory settings based on
        input data dimensions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestamps)
            Training input samples.
        y : None
            Ignored. Present for compatibility with scikit-learn pipelines.

        Returns
        -------
        self : object
            Returns the fitted transformer instance.
        """
        super().fit(X, y)
        self._check_trajectory_params(self.n_features_in_)
        return self

    def transform(self, X):
        """Transform time series into Quantum Recurrence Plots using Fidelity Kernel.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestamps)
            Input time series to transform.

        Returns
        -------
        X_transformed : ndarray
            The QRP representations of the time series.
            Shape is (n_samples, image_size, image_size) or
            (n_samples, image_size * image_size) if `flatten=True`.
            `image_size` equals the number of trajectories.
        """
        check_is_fitted(self, ['_is_fitted', '_n_trajectories', '_image_size'])

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input shape mismatch: transformer was fitted with {self.n_features_in_} "
                f"timestamps, but input has {X.shape[1]} timestamps."
            )

        X_processed = self._preprocess(X)

        n_samples = X_processed.shape[0]
        if self._image_size <= 0:
             output_shape = (n_samples, 0) if self.flatten else (n_samples, 0, 0)
             return np.zeros(output_shape)

        qrp_images = np.zeros((n_samples, self._image_size, self._image_size))

        for i in range(n_samples):
            qrp_images[i] = self._compute_qrp_matrix(X_processed[i])

        if self.flatten:
            return qrp_images.reshape(n_samples, -1)
        else:
            return qrp_images

