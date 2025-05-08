"""Preprocessing utility functions for time series."""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def segmentation(n_timestamps, n_segments, overlapping=False, **kwargs):
    """
    Segmentation function. Divides a time series of length n_timestamps into n_segments.
    
    Parameters:
    -----------
    n_timestamps : int
        Number of timestamps in the time series.
    n_segments : int
        Number of segments to divide the time series into.
    overlapping : bool, default=False
        Whether segments should overlap. Currently not implemented.
    
    Returns:
    --------
    start : numpy.ndarray
        Starting indices of each segment.
    end : numpy.ndarray
        Ending indices of each segment (exclusive).
    lengths : numpy.ndarray
        Length of each segment.
    """
    import numpy as np
    
    if not isinstance(n_timestamps, int) or n_timestamps <= 0:
        raise ValueError("n_timestamps must be a positive integer.")
    if not isinstance(n_segments, int) or n_segments <= 0:
        raise ValueError("n_segments must be a positive integer.")
    
    if n_segments > n_timestamps:
        print(f"Warning: n_segments ({n_segments}) > n_timestamps ({n_timestamps}). "
              f"Adjusting n_segments to n_timestamps.")
        n_segments = n_timestamps
    
    segment_len = n_timestamps // n_segments
    remainder = n_timestamps % n_segments
    
    lengths = np.full(n_segments, segment_len, dtype=int)
    lengths[:remainder] += 1
    
    start = np.zeros(n_segments, dtype=int)
    if not overlapping:
        start[1:] = np.cumsum(lengths[:-1])
    else:
        start[1:] = np.cumsum(lengths[:-1])
    
    end = start + lengths
    
    return start, end, lengths


def scale_time_series(ts: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Scales a single time series.

    Args:
        ts (np.ndarray): The 1D time series array.
        method (str): Scaling method ('minmax' to [0, 1], 'standard' for Z-score).

    Returns:
        np.ndarray: The scaled time series.
    Raises:
        ValueError: If the method is unknown or input is not 1D.
    """
    if ts.ndim != 1:
        if ts.ndim == 2 and (ts.shape[0] == 1 or ts.shape[1] == 1):
             ts = ts.ravel()
        else:
             raise ValueError("Input time series must be 1-dimensional.")

    ts_reshaped = ts.reshape(-1, 1)

    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    scaled_ts = scaler.fit_transform(ts_reshaped)

    return scaled_ts.flatten()