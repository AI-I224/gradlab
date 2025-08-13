"""
This PyTest module tests the methods within the Linear, ReLU and Sequential Class

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import numpy as np
from core.engine import Tensor
from core.nn import Linear, ReLU, Sequential

X = Tensor(np.random.randn(3, 5).astype(np.float32))

def test_linear_forward_shape():
    """
    Tests that the linear forward pass produces the correct dimension
    """
    layer = Linear(3, 2)  # in_features = 3, out_features = 2
    out = layer(X)
    assert out.data.shape == (2, 5), "Failed producing the correcting dimensions of output Tensor"

def test_sequential_forward_and_backward():
    """
    Tests that the multiple layers within the neural network are chain together
    such that the forward pass and backpropagation occurs
    """
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2)
    )

    out = model(X)

    loss = out.sum()
    loss.backward()

    for p in model.parameters():
        assert p.grad.shape == p.data.shape, "The multiple layers are not chained togethe correctly"
    