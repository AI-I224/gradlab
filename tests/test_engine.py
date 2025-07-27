"""
This PyTest module tests the methods within the Value Class

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import pytest
import numpy as np
from core.engine import Value

# @pytest.fixture(name="A")
# def a_value():
#     a = Value(2.0)
#     return a

# @pytest.fixture(name="B")
# def b_value():
#     b = Value(3.0)
#     return b

A = 2.0
B = 3.0
C = -4.0
D = -5.0

W = Value(A)
X = Value(B)
Y = Value(C)
Z = Value(D)

TOL = 1e07

def test_add():
    c = W + X
    assert c.data == A + B, "Operands do not add up correctly"

def test_mul():
    c = W * X
    assert c.data == A * B, "Operands do not multiply correctly"

@pytest.mark.parametrize("base, exponent", [
    (W, A),
    (W, C)
])
def test_pow(base, exponent):
    c = base ** exponent
    assert c.data == A ** exponent, "Output doesn't match the Value to the correct exponent"

def test_neg():
    c = -W
    assert c.data == -A, "Output doesn't change sign"

def test_sub():
    c = W - X
    assert c.data == A - B, "Output doesn't subtract correctly"

def test_exp():
    c = W.exp()
    assert abs(c.data - np.exp(A)) < TOL, "Output doesn't match exponential of A"

def test_tanh():
    c = W.tanh()
    assert abs(c.data - np.tanh(A)) < TOL, "Output doesn't match tanh(A)"

def test_sigmoid():
    c = W.sigmoid()
    assert abs(c.data - 1/(1 + np.exp(-A))) < TOL, "Output doesn't match sigmoid(A)"

def test_relu_greaterthanzero():
    c = W.relu()
    assert c.data == A, "Output doesn't match ReLU of A"

def test_relu_lessthanzero():
    c = Y.relu()
    assert not c.data, "Output doesn't match ReLU of C"
