"""Unit tests for QRP class"""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from qts2d.encoding.qrp import QRP

N_SAMPLES = 3
N_TIMESTAMPS = 60
DIMENSION = 3
TIME_DELAY = 2
EXPECTED_IMAGE_SIZE = N_TIMESTAMPS - (DIMENSION - 1) * TIME_DELAY 

@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    return np.random.rand(N_SAMPLES, N_TIMESTAMPS)

@pytest.fixture
def sample_data_short():
    """Generate sample time series data too short for default trajectory settings."""
    return np.random.rand(N_SAMPLES, (DIMENSION - 1) * TIME_DELAY)


def test_qrp_instantiation_defaults(sample_data):
    """Test QRP instantiation with default parameters."""
    transformer = QRP()
    assert transformer.dimension == 1
    assert transformer.time_delay == 1
    assert transformer.feature_map_reps == 2
    assert transformer.threshold is None
    assert transformer.percentage == 10
    assert not transformer.flatten
    assert transformer.scaling == 'minmax'
    
    X_transformed = transformer.fit_transform(sample_data)
    expected_default_image_size = N_TIMESTAMPS - (1 - 1) * 1
    assert X_transformed.shape == (N_SAMPLES, expected_default_image_size, expected_default_image_size)
    assert X_transformed.dtype == np.float64

def test_qrp_instantiation_custom_params(sample_data):
    """Test QRP with custom trajectory and feature map parameters."""
    transformer = QRP(dimension=DIMENSION, time_delay=TIME_DELAY, feature_map_reps=3, scaling='standard')
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE)

@pytest.mark.parametrize("threshold_val", [None, 'point', 'distance', 0.5])
@pytest.mark.parametrize("flatten_val", [True, False])
def test_qrp_thresholds_and_flatten(sample_data, threshold_val, flatten_val):
    """Test QRP with different threshold and flatten settings."""
    transformer = QRP(
        dimension=DIMENSION, time_delay=TIME_DELAY, 
        threshold=threshold_val, percentage=20, flatten=flatten_val
    )
    X_transformed = transformer.fit_transform(sample_data)
    if flatten_val:
        assert X_transformed.shape == (N_SAMPLES, EXPECTED_IMAGE_SIZE * EXPECTED_IMAGE_SIZE)
        assert X_transformed.ndim == 2
    else:
        assert X_transformed.shape == (N_SAMPLES, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE)
        assert X_transformed.ndim == 3
    
    if threshold_val is not None:
        assert np.all(np.isin(X_transformed, [0, 1]))
    else:
        assert np.all(X_transformed >= 0) and np.all(X_transformed <= 1)

def test_qrp_fit_transform_flow(sample_data):
    """Test fit and transform separately."""
    transformer = QRP(dimension=DIMENSION, time_delay=TIME_DELAY)
    transformer.fit(sample_data)
    assert transformer._is_fitted
    assert transformer._n_trajectories == EXPECTED_IMAGE_SIZE
    assert transformer._image_size == EXPECTED_IMAGE_SIZE

    X_transformed = transformer.transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE)

def test_qrp_not_fitted_error(sample_data):
    """Test that transform raises NotFittedError if called before fit."""
    transformer = QRP()
    with pytest.raises(NotFittedError):
        transformer.transform(sample_data)

def test_qrp_input_shape_mismatch_error(sample_data):
    """Test ValueError if transform is called with different n_features."""
    transformer = QRP()
    transformer.fit(sample_data)
    wrong_data = np.random.rand(N_SAMPLES, N_TIMESTAMPS + 1)
    with pytest.raises(ValueError) as excinfo:
        transformer.transform(wrong_data)
    assert "Input shape mismatch" in str(excinfo.value)

def test_qrp_trajectory_too_short_error_fit(sample_data_short):
    """Test ValueError during fit if time series is too short for trajectory settings."""
    transformer = QRP(dimension=DIMENSION, time_delay=TIME_DELAY)
    with pytest.raises(ValueError) as excinfo:
        transformer.fit(sample_data_short)
    assert "The number of trajectories" in str(excinfo.value)
    assert "must be positive" in str(excinfo.value)


@pytest.mark.parametrize("dim", [-1, 0, 0.5, "auto"])
def test_qrp_invalid_dimension(dim):
    """Test QRP with invalid dimension values."""
    with pytest.raises(ValueError if isinstance(dim, (int | float)) and dim <=0 else TypeError):
        QRP(dimension=dim)

@pytest.mark.parametrize("delay", [-1, 0, 0.5, "auto"])
def test_qrp_invalid_time_delay(delay):
    """Test QRP with invalid time_delay values."""
    with pytest.raises(ValueError if isinstance(delay, (int | float)) and delay <=0 else TypeError):
        QRP(time_delay=delay)

@pytest.mark.parametrize("reps", [-1, 0, 0.5, "auto"])
def test_qrp_invalid_feature_map_reps(reps):
    """Test QRP with invalid feature_map_reps values."""
    with pytest.raises(ValueError if isinstance(reps, (int | float)) and reps <=0 else TypeError):
        QRP(feature_map_reps=reps)

@pytest.mark.parametrize("threshold", [object(), [1,2]])
def test_qrp_invalid_threshold_type(threshold):
    """Test QRP with invalid threshold types."""
    with pytest.raises(TypeError):
        QRP(threshold=threshold)

@pytest.mark.parametrize("threshold_val", [-0.1, -10])
def test_qrp_invalid_threshold_numeric_value(threshold_val):
    """Test QRP with invalid numeric threshold values."""
    with pytest.raises(ValueError):
        QRP(threshold=threshold_val)

@pytest.mark.parametrize("percentage", [-1, 101, "auto"])
def test_qrp_invalid_percentage(percentage):
    """Test QRP with invalid percentage values."""
    with pytest.raises(ValueError if isinstance(percentage, (int | float)) else TypeError):
        QRP(percentage=percentage)

@pytest.mark.parametrize("flatten_val", [None, 1, "true"])
def test_qrp_invalid_flatten_type(flatten_val):
    """Test QRP with invalid flatten types."""
    with pytest.raises(TypeError):
        QRP(flatten=flatten_val)

def test_qrp_scaling_none(sample_data):
    """Test QRP with scaling=None."""
    transformer = QRP(scaling=None, dimension=DIMENSION, time_delay=TIME_DELAY)
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE)

def test_qrp_single_trajectory_case(sample_data):
    """Test QRP when only one trajectory can be formed."""
    single_traj_data = np.random.rand(N_SAMPLES, 5)
    transformer = QRP(dimension=3, time_delay=2)
    X_transformed = transformer.fit_transform(single_traj_data)
    assert X_transformed.shape == (N_SAMPLES, 1, 1)
    assert np.allclose(X_transformed, 0.0) # Distance to self is 0

def test_qrp_zero_trajectories_case(sample_data):
    """Test QRP when zero trajectories can be formed (should return empty)."""
    zero_traj_data = np.random.rand(N_SAMPLES, 3)
    transformer = QRP(dimension=3, time_delay=2)
    with pytest.raises(ValueError) as excinfo:
        transformer.fit(zero_traj_data)
    assert "The number of trajectories (-1) must be positive." in str(excinfo.value)

    valid_fit_data = np.random.rand(N_SAMPLES, 10)
    transformer.fit(valid_fit_data)
    with pytest.raises(ValueError) as excinfo_transform:
         transformer.transform(zero_traj_data)
    assert "Input shape mismatch" in str(excinfo_transform.value)

