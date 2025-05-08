"""Quantum Quantum Markov Transition Fields (QMTF) implementation."""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

from math import ceil, log2

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_array, check_is_fitted

from ..base_transformer import BaseQuantumTransformer
from ..utils.preprocessing import segmentation


# Adapted from pyts: https://github.com/johannfaouzi/pyts/blob/main/pyts/image/mtf.py
def _mtf_counts(X_binned, n_samples, n_timestamps, n_bins):
    """
    Computes the Markov transition matrix (counts) from binned time series.
    """
    X_mtm_counts = np.zeros((n_samples, n_bins, n_bins))
    for i in range(n_samples):
        for j in range(n_timestamps - 1):
            X_mtm_counts[i, X_binned[i, j], X_binned[i, j + 1]] += 1
    return X_mtm_counts


# Adapted from pyts: https://github.com/johannfaouzi/pyts/blob/main/pyts/image/mtf.py
def _aggregated_mtf(X_mtf, n_samples, image_size, start, end):
    """
    Aggregates the Markov Transition Field to a smaller image size.
    `start` and `end` define the segments for aggregation.
    """
    X_amtf = np.empty((n_samples, image_size, image_size))
    for i in range(n_samples):
        for j in range(image_size):
            for k in range(image_size):
                sub_matrix = X_mtf[i, start[j]:end[j], start[k]:end[k]]
                if sub_matrix.size > 0:
                    X_amtf[i, j, k] = np.mean(sub_matrix)
                else:
                    X_amtf[i, j, k] = 0.0 
    return X_amtf


class QMTF(BaseQuantumTransformer):
    """
    Quantum Markov Transition Field (QMTF).

    This implementation uses Qiskit to represent transition probabilities
    from a classically computed Markov Transition Matrix (MTM) using quantum states.
    The MTF values are then derived from these quantum states.
    Time series are first discretized into bins. Optional scaling can be applied
    before discretization. The image size can be adjusted, with aggregation
    applied if the target image size is smaller than the number of timestamps.

    Parameters
    ----------
    image_size : int or float, default=1.0
        Shape of the output images. If float, it represents a percentage
        of the size of each time series and must be between 0 and 1.
        If int, it's the exact size.
    n_bins : int, default=8
        Number of bins for discretization.
    discretizer_strategy : {'uniform', 'quantile', 'normal'}, default='quantile'
        Strategy used by KBinsDiscretizer to define the widths of the bins.
        Actual available strategies depend on whether pyts is installed.
        Common are 'uniform', 'quantile', 'normal'.
    overlapping : bool, default=False
        Parameter for the segmentation function. If True, segments may overlap
        (behavior depends on the specific segmentation function from pyts or fallback).
    flatten : bool, default=False
        If True, images are flattened to be one-dimensional.
    scaling : str or None, default='minmax'
        Time series scaling method applied *before* discretization.
        Inherited from BaseQuantumTransformer. Options: 'minmax' (scales to [-1,1]),
        'standard' (Z-score normalization), or None (no scaling).

    References
    ----------
    .. [1] Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
           Inspection and Classification Using Tiled Convolutional Neural
           Networks." AAAI Workshop (2015).
    """

    def __init__(self, image_size=1., n_bins=8,
                 discretizer_strategy='quantile',
                 overlapping=False, flatten=False,
                 scaling='minmax'):
        super().__init__(scaling=scaling)
        self.image_size = image_size
        self.n_bins = n_bins
        self.discretizer_strategy = discretizer_strategy
        self.overlapping = overlapping
        self.flatten = flatten

    def fit(self, X, y=None):
        """
        Fit the QMTF transformer.

        Validates parameters and pre-calculates image size and qubit requirements.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestamps)
            Training input samples.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        super().fit(X, y)

        self._image_size_internal_ = self._check_params(self.n_features_in_)
        self.num_qubits_for_bins_ = ceil(log2(self.n_bins))

        return self

    def transform(self, X):
        """
        Transform each time series into a Quantum MTF image.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Input data.

        Returns
        -------
        X_new : array-like
            Transformed data. Shape is (n_samples, image_size, image_size) or
            (n_samples, image_size * image_size) if flatten=True.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but transformer was fitted with "
                f"{self.n_features_in_} features."
            )

        simulator = AerSimulator(method='statevector')
        n_samples, n_timestamps = X.shape

        X_processed = self._preprocess(X)

        discretizer = KBinsDiscretizer(n_bins=self.n_bins, strategy=self.discretizer_strategy, encode='ordinal')
        X_binned = discretizer.fit_transform(X_processed.reshape(-1,1)).astype(int)
        X_binned = X_binned.reshape(1, -1)

        X_mtm_counts = _mtf_counts(
            X_binned, n_samples, n_timestamps, self.n_bins
        )
        
        X_mtm_probs = np.zeros_like(X_mtm_counts, dtype=float)
        for i in range(n_samples): 
            sample_mtm_counts = X_mtm_counts[i]
            sum_of_rows = sample_mtm_counts.sum(axis=1, keepdims=True)
            
            non_zero_rows_mask = (sum_of_rows.ravel() > 1e-9)
            if np.any(non_zero_rows_mask):
                X_mtm_probs[i][non_zero_rows_mask, :] = \
                    sample_mtm_counts[non_zero_rows_mask, :] / sum_of_rows[non_zero_rows_mask]

        X_qmtf = np.zeros((n_samples, n_timestamps, n_timestamps))
        q = self.num_qubits_for_bins_
        
        for i in range(n_samples):
            sample_mtm_prob_dist = X_mtm_probs[i] 
            sample_binned_ts = X_binned[i, :]    

            for t1 in range(n_timestamps):
                source_bin_idx = sample_binned_ts[t1]
                probs_from_source_bin = sample_mtm_prob_dist[source_bin_idx, :]

                if np.allclose(probs_from_source_bin, 0):
                    X_qmtf[i, t1, :] = 0.0
                    continue 

                desired_amplitudes = np.zeros(2**q, dtype=complex)
                sqrt_p = np.sqrt(np.maximum(probs_from_source_bin, 0))
                
                norm_sqrt_p = np.linalg.norm(sqrt_p)
                if norm_sqrt_p > 1e-9: 
                    normalized_sqrt_p = sqrt_p / norm_sqrt_p
                else: 
                    normalized_sqrt_p = sqrt_p 

                desired_amplitudes[:self.n_bins] = normalized_sqrt_p
                
                qc = QuantumCircuit(q)
                qc.initialize(desired_amplitudes, range(q))
                qc.save_statevector() 

                result = simulator.run(qc).result()
                statevector = result.get_statevector(qc)

                for t2 in range(n_timestamps):
                    target_bin_idx = sample_binned_ts[t2]
                    
                    if target_bin_idx < len(statevector.data): 
                        amplitude_of_target_bin = statevector.data[target_bin_idx]
                        probability = np.abs(amplitude_of_target_bin)**2
                        X_qmtf[i, t1, t2] = probability
                    else:
                        X_qmtf[i, t1, t2] = 0.0 
        
        window_size, remainder = divmod(n_timestamps, self._image_size_internal_)
        if remainder == 0:
            X_aqmtf = np.reshape(
                X_qmtf, 
                (n_samples, self._image_size_internal_, window_size,
                self._image_size_internal_, window_size)
            ).mean(axis=(2, 4))
        else:
            window_size += 1
            start, end, _ = segmentation(
                n_timestamps=n_timestamps,
                n_segments=self._image_size_internal_,
                overlapping=self.overlapping
            )
            
            X_aqmtf = _aggregated_mtf(
                X_qmtf, n_samples, self._image_size_internal_, start, end
            )
        
        if self.flatten:
            return X_aqmtf.reshape(n_samples, -1)
        return X_aqmtf

    def _check_params(self, n_timestamps):
        """Validate parameters and return internal image size."""
        if not isinstance(self.image_size, (int | np.integer | float | np.floating)):
            raise TypeError("'image_size' must be an integer or a float.")
        
        if isinstance(self.image_size, (int | np.integer)):
            if not (1 <= self.image_size <= n_timestamps):
                raise ValueError(
                    f"If 'image_size' is an integer ({self.image_size}), it must be >= 1 "
                    f"and <= n_timestamps ({n_timestamps})."
                )
            image_size_internal = self.image_size
        else:
            if not (0. < self.image_size <= 1.):
                raise ValueError(
                    f"If 'image_size' is a float ({self.image_size}), it must be > 0. and <= 1."
                )
            image_size_internal = ceil(self.image_size * n_timestamps)
        
        if not isinstance(self.n_bins, (int | np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not self.n_bins >= 2:
            raise ValueError(f"'n_bins' ({self.n_bins}) must be >= 2.")
        
        allowed_strategies = ['uniform', 'quantile', 'kmeans']
        if self.discretizer_strategy not in allowed_strategies:
            print(f"Warning: 'discretizer_strategy' ({self.discretizer_strategy}) "
                f"is not in the listed set {allowed_strategies}. "
                "Ensure the chosen KBinsDiscretizer backend supports it.")

        return image_size_internal
