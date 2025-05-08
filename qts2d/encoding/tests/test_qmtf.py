"""Unit tests for QMTF class"""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

from math import ceil, log2

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from qts2d.encoding.qmtf import QMTF

N_SAMPLES = 3
N_TIMESTAMPS = 100
N_BINS_DEFAULT = 8
IMAGE_SIZE_DEFAULT_RATIO = 1.0
IMAGE_SIZE_CUSTOM_INT = 32
IMAGE_SIZE_CUSTOM_FLOAT = 0.5

@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    return np.random.rand(N_SAMPLES, N_TIMESTAMPS)

@pytest.fixture
def sample_data_constant():
    """Generate constant time series data (edge case for discretization/MTM)."""
    return np.ones((N_SAMPLES, N_TIMESTAMPS)) * 0.5

def test_qmtf_instantiation_defaults(sample_data):
    """Test QMTF instantiation with default parameters."""
    transformer = QMTF()
    assert transformer.image_size == IMAGE_SIZE_DEFAULT_RATIO
    assert transformer.n_bins == N_BINS_DEFAULT
    assert transformer.discretizer_strategy == 'quantile'
    assert not transformer.overlapping
    assert not transformer.flatten
    assert transformer.scaling == 'minmax'

    X_transformed = transformer.fit_transform(sample_data)
    expected_img_size = ceil(IMAGE_SIZE_DEFAULT_RATIO * N_TIMESTAMPS)
    assert X_transformed.shape == (N_SAMPLES, expected_img_size, expected_img_size)
    assert X_transformed.dtype == np.float64

def test_qmtf_instantiation_custom_params(sample_data):
    """Test QMTF with custom image_size (int and float), n_bins, and strategy."""
    transformer_int = QMTF(image_size=IMAGE_SIZE_CUSTOM_INT, n_bins=10, discretizer_strategy='uniform', scaling='standard')
    X_transformed_int = transformer_int.fit_transform(sample_data)
    assert X_transformed_int.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM_INT, IMAGE_SIZE_CUSTOM_INT)

    transformer_float = QMTF(image_size=IMAGE_SIZE_CUSTOM_FLOAT, n_bins=5, discretizer_strategy='uniform')
    X_transformed_float = transformer_float.fit_transform(sample_data)
    expected_img_size_float = ceil(IMAGE_SIZE_CUSTOM_FLOAT * N_TIMESTAMPS)
    assert X_transformed_float.shape == (N_SAMPLES, expected_img_size_float, expected_img_size_float)

@pytest.mark.parametrize("flatten_val", [True, False])
def test_qmtf_flatten_output(sample_data, flatten_val):
    """Test QMTF with flatten=True and flatten=False."""
    transformer = QMTF(image_size=IMAGE_SIZE_CUSTOM_INT, flatten=flatten_val)
    X_transformed = transformer.fit_transform(sample_data)
    if flatten_val:
        assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM_INT * IMAGE_SIZE_CUSTOM_INT)
        assert X_transformed.ndim == 2
    else:
        assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM_INT, IMAGE_SIZE_CUSTOM_INT)
        assert X_transformed.ndim == 3

def test_qmtf_fit_transform_flow(sample_data):
    """Test fit and transform separately."""
    transformer = QMTF(image_size=IMAGE_SIZE_CUSTOM_INT, n_bins=N_BINS_DEFAULT)
    transformer.fit(sample_data)
    assert transformer._is_fitted
    assert transformer._image_size_internal_ == IMAGE_SIZE_CUSTOM_INT
    assert transformer.num_qubits_for_bins_ == ceil(log2(N_BINS_DEFAULT))

    X_transformed = transformer.transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM_INT, IMAGE_SIZE_CUSTOM_INT)

def test_qmtf_not_fitted_error(sample_data):
    """Test that transform raises NotFittedError if called before fit."""
    transformer = QMTF()
    with pytest.raises(NotFittedError):
        transformer.transform(sample_data)

def test_qmtf_input_shape_mismatch_error(sample_data):
    """Test ValueError if transform is called with different n_features."""
    transformer = QMTF()
    transformer.fit(sample_data)
    wrong_data = np.random.rand(N_SAMPLES, N_TIMESTAMPS + 10)
    with pytest.raises(ValueError) as excinfo:
        transformer.transform(wrong_data)
    assert "Input has" in str(excinfo.value)
    assert "features, but transformer was fitted with" in str(excinfo.value)

@pytest.mark.parametrize("img_size", [-1, 0, 1.1, "auto"])
def test_qmtf_invalid_image_size(img_size, sample_data):
    """Test QMTF with invalid image_size values."""
    with pytest.raises(ValueError if isinstance(img_size, (int | float)) else TypeError):
        transformer = QMTF(image_size=img_size)
        if isinstance(img_size, str):
             pass
        else:
            transformer.fit(sample_data)

@pytest.mark.parametrize("n_bins_val", [-1, 0, 1, 1.5, "auto"])
def test_qmtf_invalid_n_bins(n_bins_val, sample_data):
    """Test QMTF with invalid n_bins values."""
    with pytest.raises(ValueError if isinstance(n_bins_val, int) and n_bins_val < 2 else TypeError):
        transformer = QMTF(n_bins=n_bins_val)
        transformer.fit(sample_data)


@pytest.mark.parametrize("strategy", [123, object()])
def test_qmtf_invalid_discretizer_strategy_type(strategy, capsys):
    """Test QMTF with invalid discretizer_strategy types (warning)."""
    transformer = QMTF(discretizer_strategy=strategy)
    transformer.fit(sample_data)
    captured = capsys.readouterr()
    assert "Warning: 'discretizer_strategy'" in captured.out
    assert "is not in the listed set" in captured.out

def test_qmtf_unknown_discretizer_strategy_str(capsys, sample_data):
    """Test QMTF with an unknown discretizer_strategy string (warning)."""
    transformer = QMTF(discretizer_strategy="unknown_strategy")
    transformer.fit(sample_data) 
    captured = capsys.readouterr()
    assert "Warning: 'discretizer_strategy' ('unknown_strategy')" in captured.out
    assert "is not in the listed set" in captured.out
    with pytest.raises(ValueError):
        transformer.transform(sample_data)


def test_qmtf_scaling_none(sample_data):
    """Test QMTF with scaling=None."""
    transformer = QMTF(scaling=None, image_size=IMAGE_SIZE_CUSTOM_INT)
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM_INT, IMAGE_SIZE_CUSTOM_INT)

def test_qmtf_constant_input_series(sample_data_constant):
    """Test QMTF with a constant input time series."""
    transformer = QMTF(image_size=IMAGE_SIZE_CUSTOM_INT, n_bins=2, discretizer_strategy='uniform')
    X_transformed = transformer.fit_transform(sample_data_constant)
    assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM_INT, IMAGE_SIZE_CUSTOM_INT)

    assert not np.any(np.isnan(X_transformed))
    assert not np.any(np.isinf(X_transformed))
    assert np.allclose(X_transformed, 1.0) 

def test_qmtf_image_size_equals_n_timestamps(sample_data):
    """Test QMTF when image_size (int) is exactly n_timestamps."""
    transformer = QMTF(image_size=N_TIMESTAMPS)
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, N_TIMESTAMPS, N_TIMESTAMPS)

def test_qmtf_image_size_one(sample_data):
    """Test QMTF with image_size = 1."""
    transformer = QMTF(image_size=1)
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, 1, 1)