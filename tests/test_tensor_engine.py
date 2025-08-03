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

def test_add():
    """
    Tests that Tensor objects can be added element-wise to other Tensor objects.
    """
    c = X + Y

    d = torch.add(XT, YT)
    assert np.allclose(c.data, d.numpy(), atol=TOL), "Returns incorrect output of element-wise addition"

def test_mul():
    """
    Tests that Tensor objects can be multiplied element-wise to other Tensor objects.
    """
    c = X * Y

    d = torch.mul(XT, YT)
    assert np.allclose(c.data, d.numpy(), atol=TOL), "Returns incorrect output of element-wise multiplication"
