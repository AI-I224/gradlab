import pytest
from core.engine import Value

@pytest.fixture(name="A")
def a_value():
    a = Value(2.0)
    return a

@pytest.fixture(name="B")
def b_value():
    b = Value(3.0)
    return b

def test_addition(A, B):
    c = A + B
    assert c.data == 5.0, "Value doesn't add up to 5.0"


# def test_addition():
#     a = Value(2.0)
#     b = Value(3.0)
#     c = a + b
#     assert c.data == 5.0
