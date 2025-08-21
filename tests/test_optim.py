"""
This PyTest module tests the methods within the different optimisation algorithm subclasses

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import numpy as np
from core.engine import Tensor
from core.optim import SGD, AdaGrad, RMSProp, Adam

TOL = 1e-03

def test_sgd_step_no_momentum():
    """
    Tests the SGD optimisation step without momentum
    """
    p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    p.grad = np.array([0.5, -0.5], dtype=np.float32)

    opt = SGD([p], lr=0.1)
    opt.step()    # First step

    expected = np.array([1.0 - 0.05, 2.0 + 0.05])
    assert np.allclose(p.data, expected), "Failed optimisation step"

def test_sgd_step_with_momentum():
    """
    Tests the SGD optimisation step with momentum
    """
    p = Tensor(np.array([1.0]), requires_grad=True)
    p.grad = np.array([0.5], dtype=np.float32)

    opt = SGD([p], lr=0.1, momentum=0.9)
    opt.step()    # First step

    # velocity = -lr * grad = -0.05
    # p.data = 1.0 - 0.05 = 0.95
    assert np.allclose(p.data, np.array([0.95])), "Failed first optimisation step"

    p.grad = np.array([0.5], dtype=np.float32)
    opt.step()    # Second step with same grad

    # velocity = 0.9 * -0.05 - 0.1*0.5 = -0.045 - 0.05 = -0.095
    # p.data = 0.95 + (-0.095) = 0.855
    assert np.allclose(p.data, np.array([0.855]), rtol=TOL), "Failed second optimisation step"

def test_adagrad_step():
    """
    Tests the AdaGrad optimisation step
    """
    p = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    p.grad = np.array([0.5], dtype=np.float32)

    opt = AdaGrad([p], lr=1.0, eps=1e-8)

    opt.step()    # First step

    # g_sum[0] = 0 + (0.5)^2 = 0.25
    # update = lr * grad / sqrt(G) = 1.0 * 0.5 / sqrt(0.25) = 1.0
    # p.data = 1.0 - 1.0 = 0.0
    assert np.allclose(p.data,
                       np.array([0.0], dtype=np.float32),
                       atol=TOL),  "Failed first optimisation step"

    p.grad = np.array([0.5], dtype=np.float32)
    opt.step()    # Second step

    # g_sum[0] = 0.25 + 0.25 = 0.5
    # update = 1.0 * 0.5 / sqrt(0.5) = 0.7071 (4 d.p.)
    # p.data = 0.0 - 0.7071 = -0.7071 (4 d.p.)
    assert np.allclose(p.data,
                       np.array([-0.7071], dtype=np.float32),
                       atol=TOL),  "Failed second optimisation step"

def test_rmsprop_step():
    """
    Tests the RMSProp optimisation step
    """
    p = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    p.grad = np.array([0.5], dtype=np.float32)

    opt = RMSProp([p], lr=1.0, beta=0.9, eps=1e-8)

    opt.step()    # First step

    # v = 0.9 * 0 + 0.1 * (0.5^2) = 0.025
    # update = 1.0 * 0.5 / sqrt(0.025) = 3.1623 (4 d.p.)
    # p.data = 1.0 - 3.1623 = -2.1623 (4 d.p.)
    assert np.allclose(p.data,
                       np.array([-2.1623], dtype=np.float32),
                       atol=TOL),  "Failed first optimisation step"

    p.grad = np.array([0.5], dtype=np.float32)
    opt.step()    # Second step
    
    # v = 0.9*0.025 + 0.1*0.25 = 0.0475
    # update = 0.5 / sqrt(0.0475) = 2.295 (4 d.p.)
    # p.data = -2.1623 - 2.295 = -4.4573 (4 d.p.)
    assert np.allclose(p.data,
                       np.array([-4.4573], dtype=np.float32),
                       atol=TOL),  "Failed second optimisation step"

def test_adam_step():
    """
    Tests the Adam optimisation step
    """
    p = Tensor(np.array([1.0]), requires_grad=True)
    p.grad = np.array([0.5], dtype=np.float32)

    opt = Adam([p], lr=0.1)
    opt.step()

    # First moment, m = 0.9*0 + 0.1*0.5 = 0.05
    # Second moment, v = 0.999*0 + 0.001*0.25 = 0.00025

    # Bias correction:
    #   m_hat = 0.05 / (1-0.9^1) = 0.5
    #   v_hat = 0.00025 / (1-0.999^1) = 0.25
    
    # Update = 0.1 * (0.5 / (sqrt(0.25) + 1e-8)) = 0.1 * (0.5 / 0.5) = 0.1
    # p.data = 1.0 - 0.1 = 0.9
    assert np.allclose(p.data, np.array([0.9]), rtol=TOL), "Failed optimisation step"
