"""Quantum Spectrogram (QSG) implementation."""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

import numpy as np
import qiskit.quantum_info as qi
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from scipy.signal import get_window
from sklearn.utils.validation import check_array, check_is_fitted

from ..base_transformer import BaseQuantumTransformer


class QSG(BaseQuantumTransformer):
    """Quantum Spectrogram transformer.

    Encodes time series into spectrograms using Quantum Fourier Transform (QFT)
    on windowed segments of the time series.

    Parameters
    ----------
    segment_length : int, default=128
        Length of each segment for QFT. Must be a power of 2.
        This also determines the number of qubits used for QFT (log2(segment_length)).
    overlap_ratio : float, default=0.5
        Ratio of segment_length for overlap between consecutive segments.
        Must be in the range [0, 1). For example, 0.5 means 50% overlap.
    window_name : str or callable, default='hann'
        The window function to apply to each segment. See `scipy.signal.get_window`.
    shots : int, default=100
        Number of shots for quantum circuit execution when simulating measurements.
        Higher shots reduce variance in the estimated probabilities.
    scaling : str or None, default='minmax'
        Method to scale the input time series before segmentation and encoding.
        Inherited from `BaseQuantumTransformer`. Options: 'minmax', 'standard', None.
    flatten : bool, default=False
        If True, flatten the output spectrogram for each sample into a 1D array.
        The shape will be (n_samples, num_frequency_bins * num_time_segments).
        If False, the shape will be (n_samples, num_frequency_bins, num_time_segments).

    Attributes
    ----------
    n_features_in_ : int
        The number of features (timestamps) seen during `fit`.
    _n_qubits : int
        Number of qubits required, calculated as log2(segment_length).
    _overlap_samples : int
        Number of samples for overlap, calculated from overlap_ratio.
    _stride : int
        Step size between start of consecutive segments.
    _window : np.ndarray
        The windowing array.
    _num_freq_bins : int
        Number of frequency bins in the output spectrogram (segment_length / 2).
    _num_time_segments : int
        Number of time segments (frames) in the output spectrogram, determined during fit.
    _effective_length_for_transform : int
        The length to which input series are padded/truncated in `transform`
        to ensure a fixed number of time segments.
    _is_fitted : bool
        True if the transformer has been fitted.
    """
    def __init__(self, segment_length=128, overlap_ratio=0.5, window_name='hann',
                 shots=1024, scaling='minmax', flatten=False):
        super().__init__(scaling=scaling)
        self.segment_length = segment_length
        self.overlap_ratio = overlap_ratio
        self.window_name = window_name
        self.shots = shots
        self.flatten = flatten

        if not isinstance(self.segment_length, int) or self.segment_length <= 0 or \
           (self.segment_length & (self.segment_length - 1)) != 0:
            raise ValueError("segment_length must be a positive integer and a power of 2.")
        if not isinstance(self.overlap_ratio, (float | int)) or not (0 <= self.overlap_ratio < 1):
            raise ValueError("overlap_ratio must be a number in [0, 1).")
        if not isinstance(self.shots, int) or self.shots <= 0:
            raise ValueError("shots must be a positive integer.")
        if not isinstance(self.flatten, bool):
            raise TypeError("`flatten` must be a boolean.")

    def _get_quantum_spectrum_for_segment(self, segment_data: np.ndarray) -> np.ndarray:
        """
        Computes the 'quantum spectrum' for a single signal segment using QFT
        and simulated measurements.
        """
        if len(segment_data) != self.segment_length:
            raise ValueError(f"Segment length {len(segment_data)} must match "
                             f"transformer's segment_length {self.segment_length}.")

        norm = np.linalg.norm(segment_data)
        if np.isclose(norm, 0):
            return np.zeros(self.segment_length)

        normalized_segment = segment_data / norm

        qr = QuantumRegister(self._n_qubits, 'q')
        cr = ClassicalRegister(self._n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)

        initial_state = qi.Statevector(normalized_segment)
        qc.initialize(initial_state.data, qr)
        qc.barrier()

        qft_gate = QFT(num_qubits=self._n_qubits, approximation_degree=0, do_swaps=True,
                       inverse=False, insert_barriers=True, name='QFT')
        qc.append(qft_gate, qr)
        qc.barrier()

        qc.measure(qr, cr)

        simulator = AerSimulator(method='statevector')
        compiled_circuit = transpile(qc, simulator)
        
        result = simulator.run(compiled_circuit, shots=self.shots).result()
        counts = result.get_counts(qc)

        probabilities = np.zeros(self.segment_length)
        for i in range(self.segment_length):
            binary_representation = format(i, f'0{self._n_qubits}b')
            if binary_representation in counts:
                probabilities[i] = counts[binary_representation] / self.shots
        
        quantum_power_spectrum = probabilities * (norm**2)

        return quantum_power_spectrum


    def fit(self, X, y=None):
        """Fit the QSG transformer.

        Validates input and parameters. Determines fixed dimensions for the
        output spectrograms based on `n_features_in_` (length of time series).
        """
        super().fit(X, y) 

        self._n_qubits = int(np.log2(self.segment_length))
        self._overlap_samples = int(self.segment_length * self.overlap_ratio)
        self._stride = self.segment_length - self._overlap_samples
        if self._stride <= 0:
            raise ValueError(
                f"segment_length ({self.segment_length}) and overlap_ratio "
                f"({self.overlap_ratio}) result in a non-positive stride "
                f"({self._stride}). Decrease overlap_ratio or increase segment_length."
            )

        try:
            self._window = get_window(self.window_name, self.segment_length)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid window_name: '{self.window_name}'. Error: {e}")  # noqa: B904

        self._num_freq_bins = self.segment_length // 2

        if self.n_features_in_ < self.segment_length:
            self._num_time_segments = 1
            self._effective_length_for_transform = self.segment_length
        else:
            if self.n_features_in_ == self.segment_length and self._stride > 0 :
                 self._num_time_segments = 1
            elif self._stride == 0 and self.n_features_in_ >= self.segment_length :
                 self._num_time_segments = self.n_features_in_ - self.segment_length + 1
            else:
                 self._num_time_segments = 1 + (self.n_features_in_ - self.segment_length) // self._stride
            
            self._effective_length_for_transform = self.segment_length + (self._num_time_segments - 1) * self._stride
        
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform time series into Quantum Spectrograms."""
        check_is_fitted(self, ['_is_fitted', '_n_qubits', '_overlap_samples', 
                               '_stride', '_window', '_num_freq_bins',
                               '_num_time_segments', '_effective_length_for_transform'])
        
        X_orig = check_array(X, ensure_2d=True, dtype=np.float64)

        if X_orig.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X_orig.shape[1]} features (timestamps), but transformer "
                f"was fitted with {self.n_features_in_} features."
            )

        X_processed = self._preprocess(X_orig.copy())

        n_samples = X_processed.shape[0]
        all_spectrograms_list = []

        for i in range(n_samples):
            signal_single = X_processed[i, :]
            current_len = len(signal_single)

            if current_len < self._effective_length_for_transform:
                pad_amount = self._effective_length_for_transform - current_len
                signal_padded_final = np.pad(signal_single, (0, pad_amount), 'constant', constant_values=0)
            elif current_len > self._effective_length_for_transform:
                signal_padded_final = signal_single[:self._effective_length_for_transform]
            else:
                signal_padded_final = signal_single
            
            q_spectrogram_ts = np.zeros((self._num_freq_bins, self._num_time_segments))
            
            if self._num_time_segments == 0:
                all_spectrograms_list.append(q_spectrogram_ts)
                continue

            for k in range(self._num_time_segments):
                start = k * self._stride
                end = start + self.segment_length
                segment = signal_padded_final[start:end]

                if len(segment) != self.segment_length:
                    segment = np.pad(segment, (0, self.segment_length - len(segment)), 'constant', constant_values=0)
                
                windowed_segment = segment * self._window
                full_q_spectrum = self._get_quantum_spectrum_for_segment(windowed_segment.astype(float))
                q_spectrogram_ts[:, k] = full_q_spectrum[:self._num_freq_bins]
            
            all_spectrograms_list.append(q_spectrogram_ts)

        output_spectrograms = np.array(all_spectrograms_list)

        if self.flatten:
            return output_spectrograms.reshape(n_samples, -1)
        else:
            return output_spectrograms

