"""
This PyTest module tests the methods within the Value & Tensor Class

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import pytest
import numpy as np
from core.engine import Value

A = 2.0
B = 3.0
C = -4.0
D = -5.0

W = Value(A)
X = Value(B)
Y = Value(C)
Z = Value(D)

TOL = 1e-07 # Tolerance to check results have error smaller than 1e-07 

def test_add():
    """
    Tests that Value objects can be added to other Value objects, integers and floats.
    """
    c = W + X
    d = X + W
    e = A + X
    f = W + B
    assert c.data == d.data == e.data == f.data == A + B, "Returns incorrect output of addition"

def test_mul():
    """
    Tests that Value objects can be multiplied by other Value objects, integers and floats.
    """
    c = W * X
    d = X * W
    e = A * X
    f = W * B
    assert c.data == d.data == e.data == f.data == A * B, "Returns incorrect output of multiplication"

@pytest.mark.parametrize("base, exponent", [
    (W, A),
    (W, C)
])
def test_pow(base, exponent):
    """
    Tests that Value objects can have an exponent using integers and floats.
    """
    c = base ** exponent
    assert c.data == A ** exponent, "Returns incorrect output of exponentiation"

def test_neg():
    """
    Tests that Value objects have negative/reverse signs.
    """
    c = -W
    assert c.data == -A, "Returns incorrect sign of Value object"

def test_sub():
    """
    Tests that Value objects can be subtracted by and from other Value objects, integers and floats.
    """
    c = W - X
    d = A - X
    e = W - B
    assert c.data == d.data == e.data == A - B, "Returns incorrect output of subtraction"

def test_truediv():
    """
    Tests that Value objects can be divided by and from other Value objects, integers and floats.
    """
    c = W / X
    d = A / X
    e = W / B
    assert c.data == d.data == e.data == A / B, "Returns incorrect output of division"

def test_exp():
    """
    Tests that Value objects can be used as an exponent of e.
    """
    c = W.exp()
    assert abs(c.data - np.exp(A)) < TOL, "Returns incorrect output of exponential"

def test_tanh():
    """
    Tests that .tanh() returns the output of the Value object used in the tanh function.
    """
    c = W.tanh()
    assert abs(c.data - np.tanh(A)) < TOL, "Returns incorrect output of tanh()"

def test_sigmoid():
    """
    Tests that .sigmoid() returns the output of the Value object used in the sigmoid function.
    """
    c = W.sigmoid()
    assert abs(c.data - 1/(1 + np.exp(-A))) < TOL, "Returns incorrect output of sigmoid()"

def test_relu_greaterthanzero():
    """
    Tests that .relu() returns the output of the Value object used in the relu function
    when the Value object is greater than zero.
    """
    c = W.relu()
    assert c.data == A, "Does not return input Value"

def test_relu_lessthanzero():
    """
    Tests that .relu() returns the output of the Value object used in the relu function
    when the Value object is less than zero.
    """
    c = Y.relu()
    assert not c.data, "Does not return zero"
