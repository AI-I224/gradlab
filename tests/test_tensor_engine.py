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
    Tests that Tensor objects can be divided element-wise by other Tensor and non-Tensor objects.
    """
    c = t1 / t2

    d = torch.div(XT, YT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of element-wise division"

def test_matmul():
    """
    Tests that Tensor objects can be matrix multiplied to other Tensor objects.
    """
    c = X.matmul(Y)

    d = torch.matmul(XT, YT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of matrix multiplication"

def test_sum():
    """
    Tests that the elements of a Tensor object can be summed together.
    """
    c = X.sum()

    d = torch.sum(XT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of summing elements"

def test_mean():
    """
    Tests that the elements of a Tensor object can be averaged.
    """
    c = X.mean()

    d = torch.mean(XT)
    assert np.array_equal(c.data, d.data), "Returns incorrect mean output of elements"

def test_exp():
    """
    Tests that Value objects can be used as an exponent of e.
    """
    c = X.exp()
    d = torch.exp(XT)
    assert np.allclose(c.data, d.data, atol=TOL), "Returns incorrect output of exponential"

def test_sigmoid():
    """
    Tests that .sigmoid() returns the output of the Value object used in the sigmoid function.
    """
    c = X.sigmoid()
    d = torch.sigmoid(XT)
    assert np.allclose(c.data, d.data, atol=TOL), "Returns incorrect output of sigmoid()"

def test_tanh():
    """
    Tests that .tanh() returns the output of the Value object used in the tanh function.
    """
    # assert abs(c.data - np.tanh(A)) < TOL, "Returns incorrect output of tanh()"

def test_relu_greaterthanzero():
    """
    Tests that .relu() returns the output of the Value object used in the relu function
    when the Value object is greater than zero.
    """
    # assert c.data == A, "Does not return input Value"

def test_relu_lessthanzero():
    """
    Tests that .relu() returns the output of the Value object used in the relu function
    when the Value object is less than zero.
    """
    # assert not c.data, "Does not return zero"

