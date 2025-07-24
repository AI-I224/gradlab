import pytest
from core.engine import Value

@pytest.fixture
def a_value():
    a = Value(2.0)
    return a

@pytest.fixture
def b_value():
    b = Value(3.0)
    return b

def test_addition(a_value, b_value):
    c = a_value + b_value
    assert c.data == 5.0, "Value doesn't add up to 5.0"


# def test_addition():
#     a = Value(2.0)
#     b = Value(3.0)
#     c = a + b
#     assert c.data == 5.0
