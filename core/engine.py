import math as m

class Value:
    """
    Stores a single scalar value and its' gradient

    Attributes:
        data: The scalar number
        grad: A float value of the calculated gradient 
        _backward: A nested function to calculate the gradient
        _prev: A set of previous inputs
        _op: A string containing the operation used
    """
    def __init__(self, data, _inputs=(), _op=''):
        """
        Initializes the instance based on scalar number.

        Args:
          data: defines scalar number
          _inputs: contains previous inputs in a tuple
          _op: contains operation used between previous inputs
        """
        self.data = data
        self.grad = 0.0

        # internal attributes used for graph visualisation and debugging
        self._backward = lambda:None
        self._prev = set(_inputs)
        self._op = _op
    
    def __add__(self, other):
        """
        Returns the output of the addition between Value objects and other constants

        Args:
            other: Defines the other operand in the operation
        """

        # logic ensures that any non-Value object used in an operation
        # with a Value is also a Value
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        return out

    def __mul__(self, other):
        """
        Returns the output of the multiplication between Value objects and other constants

        Args:
            other: Defines the other operand in the expression
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        return out

    def __pow__(self, other):
        """
        Returns the output of the Value object to the power of an integer/float constant

        Args:
            other: Defines the power of the Value object
        """
        if not isinstance(other, (int, float)):
            raise TypeError("Must be an integer or float")
        out = Value(self.data ** other, (self,), f'**{other}')

        return out

    def __neg__(self):
        """
        Returns the negative of the Value object
        """
        return self * -1

    def __sub__(self, other):
        """
        Returns the output of the subtraction between Value objects and other constants

        Args:
            other: Defines the power of the Value object
        """
        return self + (-other)
    
    def sigmoid(self):
        """
        Returns the sigmoid of the Value object
        """
        out = Value(1 / (1 + (m.e ** -self.data)), (self,), 'sigmoid')

        return out

    def tanh(self):
        """
        Returns the tanh of the Value object
        """
        out = Value(2*(2*self).sigmoid().data - 1, (self,), 'tanh')

        return out
    
    def exp(self):
        """
        Returns the exponential of the Value object
        """
        out = Value(m.e ** self.data, (self,), 'exp')

        return out
    
    def relu(self):
        """
        Returns the ReLU of Value object
        """
        self.data = self.data if self.data > 0 else 0
        out = Value(self.data, (self,), 'relu')

        return out

    def __rsub__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad})"
