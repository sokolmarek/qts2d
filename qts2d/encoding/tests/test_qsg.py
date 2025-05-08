"""Unit tests for QSG class"""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from qts2d.encoding.qsg import QSG

N_SAMPLES = 4
N_TIMESTAMPS = 256
SEGMENT_LENGTH_DEFAULT = 128
SEGMENT_LENGTH_CUSTOM = 64
OVERLAP_RATIO_DEFAULT = 0.5
SHOTS_DEFAULT = 1024
TEST_SHOTS = 100 

@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    return np.random.rand(N_SAMPLES, N_TIMESTAMPS)

@pytest.fixture
def sample_data_short():
    """Generate sample time series data shorter than default segment_length."""
    return np.random.rand(N_SAMPLES, SEGMENT_LENGTH_DEFAULT // 2)


def test_qsg_instantiation_defaults(sample_data):
    """Test QSG instantiation with default parameters."""
    transformer = QSG(shots=TEST_SHOTS)
    assert transformer.segment_length == SEGMENT_LENGTH_DEFAULT
    assert transformer.overlap_ratio == OVERLAP_RATIO_DEFAULT
    assert transformer.window_name == 'hann'
    assert transformer.shots == TEST_SHOTS
    assert transformer.scaling == 'minmax'
    assert not transformer.flatten

    X_transformed = transformer.fit_transform(sample_data)
    expected_time_segments = 1 + (N_TIMESTAMPS - SEGMENT_LENGTH_DEFAULT) // (SEGMENT_LENGTH_DEFAULT - int(SEGMENT_LENGTH_DEFAULT * OVERLAP_RATIO_DEFAULT))
    expected_freq_bins = SEGMENT_LENGTH_DEFAULT // 2
    assert X_transformed.shape == (N_SAMPLES, expected_freq_bins, expected_time_segments)
    assert X_transformed.dtype == np.float64

def test_qsg_instantiation_custom_params(sample_data):
    """Test QSG with custom segment_length, overlap, window, and shots."""
    overlap = 0.25
    window = 'hamming'
    transformer = QSG(segment_length=SEGMENT_LENGTH_CUSTOM, 
                        overlap_ratio=overlap, 
                        window_name=window, 
                        shots=TEST_SHOTS, 
                        scaling='standard')
    X_transformed = transformer.fit_transform(sample_data)
    
    stride = SEGMENT_LENGTH_CUSTOM - int(SEGMENT_LENGTH_CUSTOM * overlap)
    expected_time_segments = 1 + (N_TIMESTAMPS - SEGMENT_LENGTH_CUSTOM) // stride
    expected_freq_bins = SEGMENT_LENGTH_CUSTOM // 2
    assert X_transformed.shape == (N_SAMPLES, expected_freq_bins, expected_time_segments)

@pytest.mark.parametrize("flatten_val", [True, False])
def test_qsg_flatten_output(sample_data, flatten_val):
    """Test QSG with flatten=True and flatten=False."""
    transformer = QSG(shots=TEST_SHOTS, flatten=flatten_val, segment_length=SEGMENT_LENGTH_CUSTOM)
    X_transformed = transformer.fit_transform(sample_data)

    stride = SEGMENT_LENGTH_CUSTOM - int(SEGMENT_LENGTH_CUSTOM * OVERLAP_RATIO_DEFAULT)
    expected_time_segments = 1 + (N_TIMESTAMPS - SEGMENT_LENGTH_CUSTOM) // stride
    expected_freq_bins = SEGMENT_LENGTH_CUSTOM // 2

    if flatten_val:
        assert X_transformed.shape == (N_SAMPLES, expected_freq_bins * expected_time_segments)
        assert X_transformed.ndim == 2
    else:
        assert X_transformed.shape == (N_SAMPLES, expected_freq_bins, expected_time_segments)
        assert X_transformed.ndim == 3

def test_qsg_fit_transform_flow(sample_data):
    """Test fit and transform separately."""
    transformer = QSG(shots=TEST_SHOTS, segment_length=SEGMENT_LENGTH_CUSTOM)
    transformer.fit(sample_data)
    assert transformer._is_fitted
    assert transformer._n_qubits == int(np.log2(SEGMENT_LENGTH_CUSTOM))
    assert transformer._effective_length_for_transform > 0

    X_transformed = transformer.transform(sample_data)
    stride = SEGMENT_LENGTH_CUSTOM - int(SEGMENT_LENGTH_CUSTOM * OVERLAP_RATIO_DEFAULT)
    expected_time_segments = 1 + (N_TIMESTAMPS - SEGMENT_LENGTH_CUSTOM) // stride
    expected_freq_bins = SEGMENT_LENGTH_CUSTOM // 2
    assert X_transformed.shape == (N_SAMPLES, expected_freq_bins, expected_time_segments)

def test_qsg_not_fitted_error(sample_data):
    """Test that transform raises NotFittedError if called before fit."""
    transformer = QSG(shots=TEST_SHOTS)
    with pytest.raises(NotFittedError):
        transformer.transform(sample_data)

def test_qsg_input_shape_mismatch_error(sample_data):
    """Test ValueError if transform is called with different n_features."""
    transformer = QSG(shots=TEST_SHOTS)
    transformer.fit(sample_data)
    wrong_data = np.random.rand(N_SAMPLES, N_TIMESTAMPS + 10)
    with pytest.raises(ValueError) as excinfo:
        transformer.transform(wrong_data)
    assert "Input has" in str(excinfo.value)
    assert "features (timestamps), but transformer was fitted with" in str(excinfo.value)

def test_qsg_short_input_padding_transform(sample_data_short):
    """Test QSG transform with input shorter than segment_length (after fit on longer)."""
    transformer = QSG(segment_length=SEGMENT_LENGTH_DEFAULT, shots=TEST_SHOTS)
    fit_data = np.random.rand(N_SAMPLES, SEGMENT_LENGTH_DEFAULT * 2)
    transformer.fit(fit_data)

    X_transformed = transformer.transform(sample_data_short)
    
    expected_time_segments = transformer._num_time_segments 
    expected_freq_bins = SEGMENT_LENGTH_DEFAULT // 2
    assert X_transformed.shape == (N_SAMPLES, expected_freq_bins, expected_time_segments)

def test_qsg_short_input_fit(sample_data_short):
    """Test QSG fit with input shorter than segment_length."""
    transformer = QSG(segment_length=SEGMENT_LENGTH_DEFAULT, shots=TEST_SHOTS)
    transformer.fit(sample_data_short)

    assert transformer._num_time_segments == 1
    assert transformer._effective_length_for_transform == SEGMENT_LENGTH_DEFAULT
    
    X_transformed = transformer.transform(sample_data_short)
    expected_freq_bins = SEGMENT_LENGTH_DEFAULT // 2
    assert X_transformed.shape == (N_SAMPLES, expected_freq_bins, 1)


@pytest.mark.parametrize("seg_len", [-1, 0, 100, "auto"])
def test_qsg_invalid_segment_length(seg_len):
    """Test QSG with invalid segment_length values."""
    with pytest.raises(ValueError if isinstance(seg_len, int) else TypeError):
        QSG(segment_length=seg_len, shots=TEST_SHOTS)

@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5, "auto"])
def test_qsg_invalid_overlap_ratio(overlap):
    """Test QSG with invalid overlap_ratio values."""
    with pytest.raises(ValueError if isinstance(overlap, (float | int)) else TypeError):
        QSG(overlap_ratio=overlap, shots=TEST_SHOTS)

@pytest.mark.parametrize("shots_val", [-1, 0, 0.5, "auto"])
def test_qsg_invalid_shots(shots_val):
    """Test QSG with invalid shots values."""
    with pytest.raises(ValueError if isinstance(shots_val, int) else TypeError):
        QSG(shots=shots_val)

@pytest.mark.parametrize("window", [123, object()])
def test_qsg_invalid_window_name_type(window):
    """Test QSG with invalid window_name types."""
    with pytest.raises(ValueError):
        transformer = QSG(window_name=window, shots=TEST_SHOTS)
        transformer.fit(sample_data())

def test_qsg_unknown_window_name_str(sample_data):
    """Test QSG with an unknown window_name string."""
    with pytest.raises(ValueError) as excinfo:
        transformer = QSG(window_name="unknown_window_type", shots=TEST_SHOTS)
        transformer.fit(sample_data)
    assert "Invalid window_name" in str(excinfo.value)

@pytest.mark.parametrize("flatten_val", [None, 1, "true"])
def test_qsg_invalid_flatten_type(flatten_val):
    """Test QSG with invalid flatten types."""
    with pytest.raises(TypeError):
        QSG(flatten=flatten_val, shots=TEST_SHOTS)

def test_qsg_non_positive_stride_error(sample_data):
    """Test QSG for error when stride becomes non-positive due to overlap."""
    with pytest.raises(ValueError):
        q = QSG(segment_length=32, overlap_ratio=31/32, shots=TEST_SHOTS)
        q.fit(sample_data) 

def test_qsg_zero_norm_segment(sample_data):
    """Test QSG with a segment that has zero norm (all zeros)."""
    transformer = QSG(shots=TEST_SHOTS, segment_length=SEGMENT_LENGTH_CUSTOM)
    transformer.fit(sample_data)
    zero_segment_data = np.zeros((1, N_TIMESTAMPS))
    X_transformed = transformer.transform(zero_segment_data)
    
    stride = SEGMENT_LENGTH_CUSTOM - int(SEGMENT_LENGTH_CUSTOM * OVERLAP_RATIO_DEFAULT)
    expected_time_segments = 1 + (N_TIMESTAMPS - SEGMENT_LENGTH_CUSTOM) // stride
    expected_freq_bins = SEGMENT_LENGTH_CUSTOM // 2
    
    assert X_transformed.shape == (1, expected_freq_bins, expected_time_segments)
    assert np.allclose(X_transformed, 0.0)
