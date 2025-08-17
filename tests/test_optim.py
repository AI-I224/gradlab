"""
This PyTest module tests the methods within the different optimisation algorithm subclasses

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import numpy as np
from core.engine import Tensor
from core.optim import SGD, Adam

TOL = 1e-07

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
    assert np.allclose(p.data, np.array([0.855]), rtol=TOL), "Failed second optimisation step"

def test_adam_step():
    """
    Tests the Adam optimisation step
    """
    p = Tensor(np.array([1.0]), requires_grad=True)
    p.grad = np.array([0.5], dtype=np.float32)

    opt = Adam([p], lr=0.1)
    opt.step()

    # First moment m = 0.9*0 + 0.1*0.5 = 0.05
    # Second moment v = 0.999*0 + 0.001*0.25 = 0.00025

    # Bias correction:
    #   m_hat = 0.05 / (1-0.9^1) = 0.5
    #   v_hat = 0.00025 / (1-0.999^1) = 0.25
    
    # Update = 0.1 * (0.5 / (sqrt(0.25) + 1e-8)) = 0.1 * (0.5 / 0.5) = 0.1
    # New value = 1.0 - 0.1 = 0.9
    assert np.allclose(p.data, np.array([0.9]), rtol=TOL), "Failed optimisation step"
