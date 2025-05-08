"""Unit tests for QGAF class"""

# Author: Marek Sokol <marek.sokol@cvut.cz>
# License: BSD-3-Clause

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from qts2d.encoding.qgaf import QGAF

N_SAMPLES = 5
N_TIMESTAMPS = 50
IMAGE_SIZE_DEFAULT = N_TIMESTAMPS
IMAGE_SIZE_CUSTOM = 32

@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    return np.random.rand(N_SAMPLES, N_TIMESTAMPS)

@pytest.fixture
def sample_data_short():
    """Generate sample time series data shorter than custom image size for PAA test."""
    return np.random.rand(N_SAMPLES, IMAGE_SIZE_CUSTOM // 2)

def test_qgaf_instantiation_defaults(sample_data):
    """Test QGAF instantiation with default parameters."""
    transformer = QGAF()
    assert transformer.image_size is None
    assert transformer.method == 'summation'
    assert transformer.computation_method == 'hadamard'
    assert transformer.scaling == 'minmax'
    
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_DEFAULT, IMAGE_SIZE_DEFAULT)
    assert X_transformed.dtype == np.float64

def test_qgaf_instantiation_custom_image_size(sample_data):
    """Test QGAF with a custom image_size."""
    transformer = QGAF(image_size=IMAGE_SIZE_CUSTOM)
    assert transformer.image_size == IMAGE_SIZE_CUSTOM
    
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM, IMAGE_SIZE_CUSTOM)

def test_qgaf_paa_with_shorter_input(sample_data_short):
    """Test QGAF with PAA when input is shorter than image_size (PAA should still work)."""
    transformer = QGAF(image_size=IMAGE_SIZE_CUSTOM)
    X_transformed = transformer.fit_transform(sample_data_short)
    assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM, IMAGE_SIZE_CUSTOM)


@pytest.mark.parametrize("method", ['summation', 'difference'])
@pytest.mark.parametrize("computation_method", ['hadamard', 'xu'])
def test_qgaf_methods_and_computation(sample_data, method, computation_method):
    """Test different methods and computation_methods."""
    scaling = 'minmax' if computation_method == 'hadamard' else None
    transformer = QGAF(method=method, computation_method=computation_method, scaling=scaling)
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_DEFAULT, IMAGE_SIZE_DEFAULT)

def test_qgaf_scaling_options(sample_data):
    """Test QGAF with different scaling options, particularly for 'xu'."""
    transformer_hadamard_std = QGAF(computation_method='hadamard', scaling='standard')
    transformer_hadamard_std.fit_transform(sample_data)

    transformer_xu_none = QGAF(computation_method='xu', scaling=None)
    X_transformed_xu_none = transformer_xu_none.fit_transform(sample_data)
    assert X_transformed_xu_none.shape == (N_SAMPLES, IMAGE_SIZE_DEFAULT, IMAGE_SIZE_DEFAULT)

    transformer_xu_std = QGAF(computation_method='xu', scaling='standard')
    X_transformed_xu_std = transformer_xu_std.fit_transform(sample_data)
    assert X_transformed_xu_std.shape == (N_SAMPLES, IMAGE_SIZE_DEFAULT, IMAGE_SIZE_DEFAULT)

def test_qgaf_flatten_output(sample_data):
    """Test QGAF with flatten=True (though QGAF doesn't have flatten, this tests base)."""
    transformer = QGAF()
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.ndim == 3 

def test_qgaf_fit_transform_flow(sample_data):
    """Test fit and transform separately."""
    transformer = QGAF(image_size=IMAGE_SIZE_CUSTOM)
    transformer.fit(sample_data)
    assert transformer._is_fitted
    assert transformer._image_size == IMAGE_SIZE_CUSTOM
    if IMAGE_SIZE_CUSTOM == N_TIMESTAMPS:
        assert transformer._paa is None
    else:
        assert transformer._paa is not None

    X_transformed = transformer.transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM, IMAGE_SIZE_CUSTOM)

def test_qgaf_not_fitted_error(sample_data):
    """Test that transform raises NotFittedError if called before fit."""
    transformer = QGAF()
    with pytest.raises(NotFittedError):
        transformer.transform(sample_data)

def test_qgaf_input_shape_mismatch_error(sample_data):
    """Test ValueError if transform is called with different n_features."""
    transformer = QGAF()
    transformer.fit(sample_data)
    wrong_data = np.random.rand(N_SAMPLES, N_TIMESTAMPS + 1)
    with pytest.raises(ValueError) as excinfo:
        transformer.transform(wrong_data)
    assert "Input shape mismatch" in str(excinfo.value)

@pytest.mark.parametrize("image_size", [-1, 0, "auto"])
def test_qgaf_invalid_image_size_type(image_size, sample_data):
    """Test QGAF with invalid image_size types (QGAF itself doesn't validate this in init)."""
    if isinstance(image_size, str):
        with pytest.raises(TypeError):
            transformer = QGAF(image_size=image_size)
            transformer.fit_transform(sample_data)
    elif image_size <=0 :
         with pytest.raises(ValueError):
            transformer = QGAF(image_size=image_size)
            transformer.fit_transform(sample_data)


def test_qgaf_invalid_method(sample_data):
    """Test QGAF with an invalid method."""
    with pytest.raises(ValueError) as excinfo:
        QGAF(method="invalid_method")
    assert "method must be 'summation' or 'difference'" in str(excinfo.value)

def test_qgaf_invalid_computation_method(sample_data):
    """Test QGAF with an invalid computation_method."""
    with pytest.raises(ValueError) as excinfo:
        QGAF(computation_method="invalid_computation")
    assert "computation_method must be 'hadamard' or 'xu'" in str(excinfo.value)

def test_qgaf_hadamard_scaling_warning(sample_data, capsys):
    """Test warning for hadamard method without minmax scaling."""
    transformer = QGAF(computation_method='hadamard', scaling='standard')
    transformer.fit_transform(sample_data)
    captured = capsys.readouterr()
    assert "Warning: computation_method='hadamard' typically requires scaling='minmax'" in captured.out

def test_qgaf_xu_method_direct_values(sample_data):
    """Test QGAF 'xu' method to ensure it can run with various scalings including None."""
    ts = QGAF(computation_method='xu', scaling=None)
    res_none = ts.fit_transform(sample_data)
    assert res_none.shape == (N_SAMPLES, N_TIMESTAMPS, N_TIMESTAMPS)

    ts_minmax = QGAF(computation_method='xu', scaling='minmax')
    res_minmax = ts_minmax.fit_transform(sample_data)
    assert res_minmax.shape == (N_SAMPLES, N_TIMESTAMPS, N_TIMESTAMPS)
    
    if not np.allclose(res_none, res_minmax):
        pass

    ts_xu_resized = QGAF(image_size=IMAGE_SIZE_CUSTOM, computation_method='xu', scaling='minmax')
    res_xu_resized = ts_xu_resized.fit_transform(sample_data)
    assert res_xu_resized.shape == (N_SAMPLES, IMAGE_SIZE_CUSTOM, IMAGE_SIZE_CUSTOM)

