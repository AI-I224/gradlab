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
            other: 
        """

        # logic ensures that any non-Value object used in an operation
        # with a Value is also a Value
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        return out

    def __mul__(self, other):
        pass

    def __pow__(self, other):
        pass

    def __neg__(self):
        pass

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad})"
