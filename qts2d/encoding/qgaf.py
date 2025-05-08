"""Quantum Gramian Angular Fields (QGAF) implementation."""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from sklearn.utils.validation import check_array, check_is_fitted

from ..base_transformer import BaseQuantumTransformer


class QGAF(BaseQuantumTransformer):
    """Quantum Gramian Angular Field transformer.

    Encodes time series into images using quantum approaches inspired by GAF.
    Supports two computation methods:
    1. 'hadamard': Scales data to [-1, 1], computes angles (phi), then uses the
       Hadamard Test.
    2. 'xu': Operates directly on the (optionally scaled) time series
       without computing intermediate angles. Implementation according to
       Xu et al. [1].

    If `image_size` is provided and differs from the input time series length,
    Piecewise Aggregate Approximation (PAA) is used for resizing *before*
    encoding.

    Parameters
    ----------
    image_size : int or None, default=None
        The desired size of the output image (image_size x image_size).
        If None, it defaults to the length of the input time series.
        If set to a value different from the input time series length,
        PAA will be used to resize the series before encoding.
    method : {'summation', 'difference'}, default='summation'
        Type of Quantum Gramian Angular Field: 'summation' (QGASF) or 
        'difference' (QGADF). Determines the relationship computed 
        between time points.
    computation_method : {'hadamard', 'xu'}, default='hadamard'
        The method used to compute the QGAF matrix.
        - 'hadamard': Uses scaling to [-1, 1], calculates angles (phi), and
          computes trigonometric sum/diff via Hadamard test. Requires `scaling`
          that results in the [-1, 1] range (e.g., 'minmax').
        - 'xu': Computes the matrix directly from the time series values
          (after potential PAA and optional scaling). Does not compute phi.
          Method according to Xu et al. [1].
    scaling : str or None, default='minmax'
        Time series scaling method (inherited from BaseQuantumTransformer).
        - If `computation_method='angle'`, this *must* scale to [-1, 1]
          (e.g., 'minmax', which is the default).
        - If `computation_method='direct'`, scaling is optional (`None`) or
          can use other methods like 'standard'.

    References
    ----------
    .. [1] Xu, Z., Wang, Y., Feng, X., Wang, Y., Li, Y., & Lin, H. (2024). 
    Quantum-enhanced forecasting: Leveraging quantum gramian angular field 
    and CNNs for stock return predictions. Finance Research Letters, 67, 105840. 
    https://doi.org/10.1016/j.frl.2024.105840

    Examples
    ----------
    TODO: Add examples for usage.
    
    """
    def __init__(self, image_size=None, method='summation', computation_method='hadamard', scaling='minmax'):
        super().__init__(scaling=scaling)
        self.image_size = image_size
        self.method = method
        self.computation_method = computation_method
        
        if method not in ['summation', 'difference']:
            raise ValueError("method must be 'summation' or 'difference'")
        if computation_method not in ['hadamard', 'xu']:
            raise ValueError("computation_method must be 'hadamard' or 'xu'")
        if computation_method == 'hadamard' and scaling != 'minmax':
            print(f"Warning: computation_method='hadamard' typically requires scaling='minmax' "
                  f"to ensure data is in [-1, 1] for arccos. Current scaling: '{scaling}'.")


    def _compute_phi(self, ts_scaled):
        """
        Takes a time series scaled to [-1, 1] and computes angles phi.

        Args:
            ts_scaled (np.ndarray): 1D array scaled to [-1, 1].

        Returns:
            np.ndarray: Array of angles (phi) in radians.
        """
        ts_clipped = np.clip(ts_scaled, -1.0, 1.0)
        phi = np.arccos(ts_clipped)
        return phi


    def _compute_qgaf_matrix_xu(self, ts):
        """
        Computes the GAF-like matrix directly from the time series segment `ts`.
        This method avoids calculating angles (phi).

        Args:
            ts (np.ndarray): 1D array representing a time series segment.

        Returns:
            np.ndarray: The resulting n x n quantum GAF matrix.
        """
        n = len(ts)
        qgaf_matrix = np.zeros((n, n))
        pauli_z = Operator.from_label('Z')
        difference = (self.method == 'difference')

        for i in range(n):
            for j in range(n):
                a = ts[i]
                b = ts[j]

                qc = QuantumCircuit(1)

                qc.ry(2 * a, 0)
                if difference:
                    qc.ry(-2 * b, 0)
                else:
                    qc.ry(2 * b, 0)

                state = Statevector.from_instruction(qc)
                expectation_Z = state.expectation_value(pauli_z)

                cos_2ab = expectation_Z.real
                
                qgaf_matrix[i, j] = cos_2ab

        return qgaf_matrix


    def _compute_qgaf_matrix_hadamard(self, phi):
        """
        Computes the GAF matrix using the Hadamard Test quantum circuit from angles.
        Calculates cos(phi_i + phi_j) or sin(phi_i - phi_j).

        Args:
            phi (np.ndarray): Array of angles computed from the time series.

        Returns:
            np.ndarray: The resulting n x n matrix.
        """
        n = len(phi)
        qgaf_matrix = np.zeros((n, n))
        pauli_z = Operator.from_label('Z')
        pauli_y = Operator.from_label('Y')

        difference = (self.method == 'difference')

        for i in range(n):
            for j in range(n):
                a = phi[i]
                b = phi[j]

                qc = QuantumCircuit(2)

                if difference:
                    theta = 2 * (a - b)
                    qc.h(0)
                    qc.crz(theta, 0, 1)
                    qc.h(0)
                else:
                    theta = 2 * (a + b)
                    qc.h(0)
                    qc.cry(theta, 0, 1)
                    qc.h(0)

                #qc.save_statevector()
                state = Statevector.from_instruction(qc)
                
                if difference:
                    expectation_Z_ancilla = state.expectation_value(pauli_y, qargs=[0])
                else:
                    expectation_Z_ancilla = state.expectation_value(pauli_z, qargs=[0])

                qgaf_matrix[i, j] = expectation_Z_ancilla.real
                        
        return qgaf_matrix


    def fit(self, X, y=None):
        """Fit the QGAF transformer.

        Validates input, and parameters. Determines image size and PAA setup.

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

        if self.image_size is None:
            self._image_size = self.n_features_in_
            self._paa = None
        else:
            self._image_size = self.image_size
            if self._image_size == self.n_features_in_:
                self._paa = None
            else:
                self._paa = PiecewiseAggregateApproximation(window_size=None, output_size=self._image_size)

        return self


    def transform(self, X):
        """Transform time series into Quantum Gramian Angular Fields.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestamps)
            Input time series.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, image_size, image_size)
            The QGAF representations of the time series.
        """
        check_is_fitted(self, '_is_fitted')
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Input shape mismatch: expected {self.n_features_in_} features, got {X.shape[1]}")

        X_resized = self._paa.transform(X) if self._paa is not None else X

        n_samples = X_resized.shape[0]
        qgaf_images = np.zeros((n_samples, self._image_size, self._image_size))

        for i in range(n_samples):
            ts_resized = X_resized[i]

            if self.computation_method == 'hadamard':
                ts_scaled = self._preprocess(ts_resized.reshape(1, -1)).flatten()
                phi = self._compute_phi(ts_scaled)
                qgaf_images[i] = self._compute_qgaf_matrix_hadamard(phi)
            elif self.computation_method == 'xu':
                if self.scaling is not None:
                    ts_processed = self._preprocess(ts_resized.reshape(1, -1)).flatten()
                else:
                    ts_processed = ts_resized
                qgaf_images[i] = self._compute_qgaf_matrix_xu(ts_processed)

        return qgaf_images
