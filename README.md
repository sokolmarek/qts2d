<p align="center">
  <img width="200" height="200" src="https://i.postimg.cc/9Q6GyVZ2/qts2d-logo.png">
</p>

# QTS2D: Quantum-Based Image Encoding of Time Series

[![PyPI version](https://img.shields.io/pypi/v/qts2d)](https://pypi.python.org/pypi/qts2d)
[![PyPI wheel](https://img.shields.io/pypi/wheel/qts2d)](https://pypi.python.org/pypi/qts2d)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/qts2d)](https://pypi.python.org/pypi/qts2d)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qts2d.svg)](https://pypi.python.org/pypi/qts2d)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**QTS2D** is a Python library for encoding time series data into image representations using quantum computing principles. These image-based representations can then be used as input for various machine learning models, particularly Convolutional Neural Networks (CNNs), for tasks like time series classification or forecasting.

## Description

The library provides several quantum-inspired transformers that convert 1D time series into 2D matrices (images). These methods are analogous to classical time series imaging techniques but leverage quantum computations or quantum-inspired mathematical formulations.

Currently implemented transformers:

*   **QGAF**: Quantum Gramian Angular Field
*   **QRP**: Quantum Recurrence Plot
*   **QSG**: Quantum Spectrogram
*   **QMTF**: Quantum Markov Transition Field

## Installation

You can install library using pip:

```bash
pip install qts2d
```

or using pip+git for the latest version of the code:

```bash
pip install git+https://github.com/sokolmarek/qts2d
```

The library requires Python >= 3.10 and the following main dependencies:

*   numpy (>=2.2.5)
*   scikit-learn (>=1.6.1)
*   qiskit (==1.4.2)
*   qiskit-aer (==0.17.0)
*   qiskit-machine-learning (==0.8.2)
*   pyts (>=0.13.0)

## Quick Example

Each transformer follows the scikit-learn `Estimator` and `TransformerMixin` interface.

```python
import numpy as np
from qts2d.encoding import QGAF

# Example time series data (n_samples, n_timestamps)
X = np.random.rand(1, 1000)

# Using Quantum Gramian Angular Field (QGAF)
qgaf = QGAF(image_size=32, method='summation', computation_method='hadamard', scaling='minmax')
X_qgaf = qgaf.fit_transform(X)
print(f"QGAF output shape: {X_qgaf.shape}")
```

For more detailed examples, please refer to the `examples/` directory for each specific transformer. To run the examples Matplotlib is required.

## Contributing
We welcome everyone to contribute to ```qts2d```! The library is still evolving and far from perfect, so your help is especially valuable. Please feel free to submit a pull request or open an issue. Detailed contribution guidelines will be shared soon along with the documentation. Stay tuned!

## Citation
Coming soon!

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

## References

* [1] Xu, Z., Wang, Y., Feng, X., Wang, Y., Li, Y., & Lin, H.
    Quantum-enhanced forecasting: Leveraging quantum gramian angular field 
    and CNNs for stock return predictions. Finance Research Letters (2024)

* [2] Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
    Inspection and Classification Using Tiled Convolutional Neural
    Networks." AAAI Workshop (2015).

* [3] J.-P Eckmann, S. Oliffson Kamphorst and D Ruelle, "Recurrence
    Plots of Dynamical Systems". Europhysics Letters (1987).