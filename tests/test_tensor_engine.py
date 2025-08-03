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
B = [[2, 0], [1, 2]]

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
    Tests that Tensor objects can be added element-wise to other Tensor objects.
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
    Tests that Tensor objects can be multiplied element-wise to other Tensor objects.
    """
    c = t1 * t2

    d = torch.mul(XT, YT)
    assert np.array_equal(c.data, d.data), "Returns incorrect output of element-wise multiplication"
