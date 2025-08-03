"""
This PyTest module tests the methods within the Tensor Class

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import pytest
import numpy as np
import torch
from core.engine import Tensor

A = [[1, 2], [3, 4]]
B = [[2, 6], [1, 2]]
C = 2
D = -3

X = Tensor(A)
Y = Tensor(B)

XT = torch.Tensor(A)
YT = torch.Tensor(B)

TOL = 1e-07 # Tolerance to check results have error smaller than 1e-07 

@pytest.mark.parametrize("t1, t2", [
    (X, Y),
    (A, Y),
    (X, B)
])
def test_add(t1, t2):
    """
    Tests that Tensor objects can be added element-wise to other Tensor and non-Tensor objects.
    """
    c = t1 + t2
    
    d = torch.add(XT, YT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of element-wise addition"

@pytest.mark.parametrize("t1, t2", [
    (X, Y),
    (A, Y),
    (X, B)
])
def test_mul(t1, t2):
    """
    Tests that Tensor objects can be multiplied element-wise to other Tensor and non-Tensor objects.
    """
    c = t1 * t2

    d = torch.mul(XT, YT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of element-wise multiplication"

@pytest.mark.parametrize("base, exponent", [
    (X, C),
    (X, D)
])
def test_pow(base, exponent):
    """
    Tests that Tensor objects can have an exponent using integers and floats.
    """
    c = base ** exponent

    d = torch.pow(XT, exponent)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of element-wise exponentiation"

def test_neg():
    """
    Tests that Tensor objects have negative/reverse signs element-wise.
    """
    c = -X
    d = torch.neg(XT)
    assert np.array_equal(c.data, d.data), "Returns incorrect sign of Tensor object's elements"

@pytest.mark.parametrize("t1, t2", [
    (X, Y),
    (A, Y),
    (X, B)
])
def test_sub(t1, t2):
    """
    Tests that Tensor objects can be subtracted element-wise
    by and from other Tensor and non-Tensor objects.
    """
    c = t1 - t2

    d = torch.sub(XT, YT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of element-wise subtraction"

@pytest.mark.parametrize("t1, t2", [
    (X, Y),
    (A, Y),
    (X, B)
])
def test_true_div(t1, t2):
    """
    Tests that Tensor objects can be multiplied element-wise to other Tensor and non-Tensor objects.
    """
    c = t1 / t2

    d = torch.div(XT, YT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of element-wise multiplication"
