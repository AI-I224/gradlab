"""
This PyTest module tests the methods within the different optimisation algorithm subclasses

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import numpy as np
from core.engine import Tensor
from core.optim import SGD

def test_sgd_step_no_momentum():
    """
    Tests the SGD optimisation step without momentum
    """
    p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    p.grad = np.array([0.5, -0.5], dtype=np.float32)

    opt = SGD([p], lr=0.1)
    opt.step()

    expected = np.array([1.0 - 0.05, 2.0 + 0.05])
    assert np.allclose(p.data, expected), "Failed optimisation step"

def test_sgd_step_with_momentum():
    """
    Tests the SGD optimisation step with momentum
    """
    p = Tensor(np.array([1.0]), requires_grad=True)
    p.grad = np.array([0.5], dtype=np.float32)

    opt = SGD([p], lr=0.1, momentum=0.9)
    opt.step()    # First update

    # velocity = -lr * grad = -0.05
    # p.data = 1.0 - 0.05 = 0.95
    assert np.allclose(p.data, np.array([0.95])), "Failed first optimisation step"

    p.grad = np.array([0.5], dtype=np.float32)
    opt.step()    # Second update with same grad

    # velocity = 0.9 * -0.05 - 0.1*0.5 = -0.045 - 0.05 = -0.095
    # p.data = 0.95 + (-0.095) = 0.855
    assert np.allclose(p.data, np.array([0.855]), rtol=1e-5), "Failed second optimisation step"
