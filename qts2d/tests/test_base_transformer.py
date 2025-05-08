import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted

from qts2d.base_transformer import BaseQuantumTransformer

N_SAMPLES = 2
N_TIMESTAMPS = 20

@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    return np.random.rand(N_SAMPLES, N_TIMESTAMPS)

class DummyTransformer(BaseQuantumTransformer):
    """A minimal subclass for testing BaseQuantumTransformer's direct functionalities."""
    def __init__(self, scaling='minmax', custom_param=None):
        super().__init__(scaling=scaling)
        self.custom_param = custom_param

    def fit(self, X, y=None):
        super().fit(X,y)
        self.dummy_fitted_attr_ = True 
        return self

    def transform(self, X):
        check_is_fitted(self, 'dummy_fitted_attr_')
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but transformer was fitted with "
                f"{self.n_features_in_} features."
            )
        X_processed = self._preprocess(X)
        return X_processed.reshape(X_processed.shape[0], X_processed.shape[1], 1)

def test_base_transformer_instantiation():
    """Test BaseQuantumTransformer instantiation via a dummy subclass."""
    transformer = DummyTransformer(scaling='standard', custom_param='test')
    assert transformer.scaling == 'standard'
    assert transformer.custom_param == 'test'
    assert hasattr(transformer, 'fit')
    assert hasattr(transformer, 'transform')
    assert hasattr(transformer, '_preprocess')

def test_base_transformer_fit(sample_data):
    """Test the fit method of BaseQuantumTransformer."""
    transformer = DummyTransformer()
    transformer.fit(sample_data)
    assert transformer._is_fitted
    assert transformer.n_features_in_ == N_TIMESTAMPS

def test_base_transformer_transform_not_fitted(sample_data):
    """Test that transform raises NotFittedError if called before fit."""
    transformer = DummyTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(sample_data)

def test_base_transformer_transform_input_shape_mismatch(sample_data):
    """Test ValueError if transform is called with different n_features."""
    transformer = DummyTransformer()
    transformer.fit(sample_data)
    wrong_data = np.random.rand(N_SAMPLES, N_TIMESTAMPS + 1)
    with pytest.raises(ValueError) as excinfo:
        transformer.transform(wrong_data)
    assert "Input has" in str(excinfo.value)
    assert "features, but transformer was fitted with" in str(excinfo.value)

@pytest.mark.parametrize("scaling_method", ['minmax', 'standard', None])
def test_base_transformer_preprocess_methods(sample_data, scaling_method):
    """Test different scaling methods in _preprocess."""
    transformer = DummyTransformer(scaling=scaling_method)
    transformer.fit(sample_data)
    
    processed_data = transformer._preprocess(sample_data.copy()) 

    assert processed_data.shape == sample_data.shape
    if scaling_method == 'minmax':
        assert np.all(processed_data >= -1 - 1e-9) and np.all(processed_data <= 1 + 1e-9)
    elif scaling_method == 'standard':
        for i in range(processed_data.shape[0]):
            assert np.isclose(np.mean(processed_data[i]), 0, atol=1e-9)
            assert np.isclose(np.std(processed_data[i]), 1, atol=1e-9) or np.isclose(np.std(processed_data[i]), 0, atol=1e-9)
    elif scaling_method is None:
        assert np.array_equal(processed_data, sample_data)

def test_base_transformer_preprocess_unknown_method(sample_data):
    """Test _preprocess with an unknown scaling method."""
    transformer = DummyTransformer(scaling='unknown_scaling')
    with pytest.raises(ValueError) as excinfo:
        transformer._preprocess(sample_data)
    assert "Unknown scaling method: unknown_scaling" in str(excinfo.value)

def test_base_transformer_fit_transform_flow(sample_data):
    """Test the full fit_transform flow using the dummy transformer."""
    transformer = DummyTransformer(scaling='minmax')
    X_transformed = transformer.fit_transform(sample_data)
    assert X_transformed.shape == (N_SAMPLES, N_TIMESTAMPS, 1)
    original_data_scaled_expected = np.array([scale_time_series(ts, method='minmax') for ts in sample_data])
    assert np.allclose(X_transformed.reshape(N_SAMPLES, N_TIMESTAMPS), original_data_scaled_expected)

def scale_time_series(ts: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Scales a single time series. Copied for direct comparison if needed."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

