"""
This PyTest module tests the methods within the Linear, ReLU and Sequential Class

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import numpy as np
import pytest
from core.engine import Tensor
from core.nn import Linear

def test_linear_forward_shape():
    """
    Tests that the linear forward pass produces the correct dimension
    """
    layer = Linear(3, 2)  # in_features=3, out_features=2
    x = Tensor(np.random.randn(3, 4).astype(np.float32))  # (features, batch_size) = (3, 4)
    out = layer(x)
    assert out.data.shape == (2, 4)  # Output: (out_features, batch_size)
    