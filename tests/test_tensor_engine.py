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

X = Tensor(A, requires_grad=True)
Y = Tensor(B, requires_grad=True)

XT = torch.Tensor(A)
XT.requires_grad = True
YT = torch.Tensor(B)
YT.requires_grad = True

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
    assert c.data == d.data, "Returns incorrect output of summing elements"

def test_mean():
    """
    Tests that the elements of a Tensor object can be averaged.
    """
    c = X.mean()

    d = torch.mean(XT)
    assert c.data == d.data, "Returns incorrect mean output of elements"

def test_exp():
    """
    Tests that Tensor objects can be used as an exponent of e.
    """
    c = X.exp()
    d = torch.exp(XT)
    assert np.allclose(c.data, d.data, atol=TOL), "Returns incorrect output of exponential"

def test_sigmoid():
    """
    Tests that .sigmoid() returns the output of the Tensor object used in the sigmoid function.
    """
    c = X.sigmoid()
    d = torch.sigmoid(XT)
    assert np.allclose(c.data, d.data, atol=TOL), "Returns incorrect output of sigmoid()"

def test_tanh():
    """
    Tests that .tanh() returns the output of the Tensor object used in the tanh function.
    """
    c = X.tanh()
    d = torch.tanh(XT)
    assert np.allclose(c.data, d.data, atol=TOL), "Returns incorrect output of tanh()"

def test_relu():
    """
    Tests that .relu() returns the output of the Tensor object used in the ReLU function.
    """
    c = X.relu()
    d = torch.relu(XT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of relu()"

def test_backward_add():
    """
    Tests backpropagation of element-wise addition operation
    """
    c = X + Y
    c.backward()

    y = XT + YT
    y.sum().backward()
    
    assert np.array_equal(X.grad, XT.grad), "Failed backpropagation for element-wise addition"
    c.zero_grad() # Reset gradients
    XT.grad = None
    YT.grad = None

def test_backward_mul():
    """
    Tests backpropagation of element-wise multiplication operation
    """
    c = X * Y
    c.backward()

    y = XT * YT
    y.sum().backward()
    
    assert np.array_equal(X.grad, XT.grad), "Failed backpropagation for element-wise multiplication"
    c.zero_grad()
    XT.grad = None
    YT.grad = None

def test_backward_pow():
    """
    Tests backpropagation of element-wise exponentiation operation
    """
    c = X ** 2
    c.backward()

    y = XT ** 2
    y.sum().backward()
    
    assert np.array_equal(X.grad, XT.grad), "Failed backpropagation for element-wise exponentiation"

def test_backward_matmul():
    """
    Tests backpropagation of matrix multiplication
    """
    c = X.matmul(Y)
    c.backward()

    y = XT.matmul(YT)
    y.sum().backward()
    
    assert np.array_equal(X.grad, XT.grad), "Failed backpropagation for matrix multiplication"

def test_backward_sum():
    """
    Tests backpropagation of the sum of all elements inside a Tensor
    """
    c = X.sum()
    c.backward()

    y = XT.sum()
    y.backward()
    
    assert np.array_equal(X.grad, XT.grad), "Failed backpropagation for summing Tensor elements"

def test_backward_mean():
    """
    Tests backpropagation of the mean of all elements inside a Tensor
    """
    c = X.mean()
    c.backward()

    y = XT.mean()
    y.backward()
    
    assert np.array_equal(X.grad, XT.grad), "Failed backpropagation for averaging Tensor elements"

def test_backward_exp():
    """
    Tests backpropagation of the exponential function
    """
    c = X.exp()
    c.backward()

    y = torch.exp(XT)
    y.sum().backward()
    
    assert np.allclose(X.grad, XT.grad, atol=TOL), "Failed backpropagation for exponential"

def test_backward_sigmoid():
    """
    Tests backpropagation of the sigmoid function
    """
    c = X.sigmoid()
    c.backward()

    y = torch.sigmoid(XT)
    y.sum().backward()
    
    assert np.allclose(X.grad, XT.grad, atol=TOL), "Failed backpropagation for sigmoid"
    