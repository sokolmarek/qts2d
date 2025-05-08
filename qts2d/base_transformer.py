"""Base classes for QTS2D transformers."""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .utils.preprocessing import scale_time_series


class BaseQuantumTransformer(BaseEstimator, TransformerMixin):
    """Base class for all quantum transformers in QTS2D.

    Handles backend selection and basic input validation.

    Parameters
    ----------
    scaling : str or None, default='minmax'
        Method to scale the input time series before encoding.
        Options: 'minmax' (scales to [0, 1]), 'standard' (zero mean, unit variance),
        None (no scaling). Other scaling methods could be added.
    """
    def __init__(self, scaling='minmax'):
        self.scaling = scaling

    def _preprocess(self, X):
        """Applies scaling to the input data."""
        if self.scaling is None:
            return X
        elif self.scaling == 'minmax':
            return np.array([scale_time_series(ts, method='minmax') for ts in X])
        elif self.scaling == 'standard':
            return np.array([scale_time_series(ts, method='standard') for ts in X])
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")

    def fit(self, X, y=None):
        """Fit the transformer.

        Validates input data. Subclasses may override this
        to perform algorithm-specific setup.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestamps)
            The training input samples (time series).
        y : None
            There is no need for a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform the time series using the specific quantum encoding.

        This method must be implemented by subclasses.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestamps)
            The input samples (time series) to transform.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, image_height, image_width)
            The transformed data (image representations).
        """
        check_is_fitted(self, '_is_fitted')
        X = check_array(X, ensure_2d=True, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but transformer was fitted with "
                f"{self.n_features_in_} features."
            )

        self._preprocess(X)

        raise NotImplementedError("Transform method must be implemented by subclasses.")
