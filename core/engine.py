"""
This Python module defines the Value and Tensor Classes for building neural networks

The Value Class describes a scalar value and how it interacts with other Value objects
and non-Value numeric objects (ie integer and float)

The Tensor Class describes a N-dimensional array and how it  interacts with other Tensor objects
and non-Tensor objects (ie lists)
"""

import math as m
import numpy as np

class Value:
    """
    Stores a single scalar value and its' gradient

    Attributes:
        data: A scalar number
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
    
    def backprop(self):
        """
        Helper function to call _backward without raising Pylint
        """
        return self._backward()
    
    def inputs(self):
        """
        Helper function to call _prev without raising Pylint
        """
        return self._prev
    
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

        def _backward():
            """
            Returns the local gradient contribution 
            """
            # Multiply local gradient, dout/dself & dout/dother by incoming gradient, dL/dout
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Returns the output of a multiplication between Value objects and other constants

        Args:
            other: Defines the other operand in the expression
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            """
            Returns the local gradient contribution 
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Returns the output of a Value object to the power of an integer/float constant

        Args:
            other: Defines the power of a Value object
        """
        if not isinstance(other, (int, float)):
            raise TypeError("Must be an integer or float")
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            """
            Returns the local gradient contribution 
            """
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        """
        Returns the negative of a Value object
        """
        return self * -1

    def __sub__(self, other):
        """
        Returns the output of subtraction between Value objects and other constants

        Args:
            other: Defines the other operand in the expression
        """
        return self + (-other)

    def __truediv__(self, other):
        """
        Returns the output of a Value object divided by other Value objects, integers and floats

        Args:
            other: Defines the other operand in the expression
        """
        return self * (other ** -1)

    def __rsub__(self, other):
        """
        Returns the output of subtraction when operands are reversed

        Args:
            other: Defines the other operand in the expression
        """
        return other + (-self)

    def __radd__(self, other):
        """
        Returns the output of addition when operands are reversed

        Args:
            other: Defines the other operand in the expression
        """
        return self + other

    def __rmul__(self, other):
        """
        Returns the output of multiplication when operands are reversed

        Args:
            other: Defines the other operand in the expression
        """
        return  self * other

    def __rtruediv__(self, other):
        """
        Returns the output of division when operands are reversed

        Args:
            other: Defines the other operand in the expression
        """
        return other * (self ** -1)
     
    def exp(self):
        """
        Returns the exponential of a Value object
        """
        out = Value(m.e ** self.data, (self,), 'exp')

        def _backward():
            """
            Returns the local gradient contribution 
            """
            self.grad += (m.e ** self.data) * out.grad
        out._backward = _backward

        return out
   
    def sigmoid(self):
        """
        Returns the sigmoid of a Value object
        """
        sigmoid = 1 / (1 + (m.e ** -self.data))
        out = Value(sigmoid, (self,), 'sigmoid')

        def _backward():
            """
            Returns the local gradient contribution 
            """
            self.grad += (sigmoid * (1 - sigmoid)) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        """
        Returns the tanh of a Value object
        """
        tanh = 2*(2*self).sigmoid().data - 1
        out = Value(tanh, (self,), 'tanh')

        def _backward():
            """
            Returns the local gradient contribution 
            """
            self.grad += (1 - (tanh ** 2)) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        """
        Returns the ReLU of Value object
        """
        self.data = self.data if self.data > 0 else 0
        out = Value(self.data, (self,), 'relu')

        def _backward():
            """
            Returns the local gradient contribution 
            """
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        """
        Propagates through the gradients backward from output to input,
        updating the gradient for each Value, and returns the gradient
        of the output with respect to the selected Value

        For example:
        x = a * b
        x.backward()

        x.grad = 1
        a.grad = dy/da = b
        b.grad = dy/db = a
        """
        topo = []
        visited = set()

        # Use topological sorting to build DAG and correctly compute gradients
        def build_topo(value):
            if value not in visited:
                visited.add(value)
                for i in value.inputs():
                    build_topo(i)
                topo.append(value)
        
        build_topo(self)

        self.grad = 1 # Set the final output gradient to 1
        for value in reversed(topo):
            value.backprop()

    def zero_grad(self):
        """
        Resets the gradient of the Value object to 0
        """
        visited = set()

        def _reset(value):
            if value not in visited:
                visited.add(value)
                value.grad = 0
                for i in value.inputs():
                    _reset(i)
        
        _reset(self)

    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad})"


class Tensor:
    """
    Stores a multi-dimensional array and its' gradient, while also determining
    whether the gradient is calculated

    Attributes:
        data: A N-dimensional array
        requires_grad: A boolean that decides whether the autograd engine 
          tracks the Tensor's operations
        grad: An array of the calculated gradient
        _backward: A nested function to calculate the gradient
        _prev: A set of previous inputs
        _op: A string containing the operation used
    """
    def __init__(self, data, requires_grad=False, _inputs=(), _op=''):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_inputs)
        self._op = _op

    def __add__(self, other):
        """
        Returns the output of the element-wise addition between Tensor objects and other arrays

        Args:
            other: Defines the other operand in the operation
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     (self.requires_grad or other.requires_grad),
                     (self, other),
                     '+')

        return out

    def __mul__(self, other):
        """
        Returns the output of the element-wise multiplication between Tensor objects and other 
        non-Tensor object arrays

        Args:
            other: Defines the other operand in the operation
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     (self.requires_grad or other.requires_grad),
                     (self, other),
                     '*')

        return out
    
    def __pow__(self, other):
        """
        Returns the output of a Tensor object to the element-wise power of an integer/float constant

        Args:
            other: Defines the power of a Tensor object
        """
        if not isinstance(other, (int, float)):
            raise TypeError("Must be an integer or float")
        out = Tensor(self.data ** other,
                     self.requires_grad,
                     (self,),
                     f'**{other}')

        return out
    
    def __neg__(self):
        """
        Returns the element-wise negative of a Tensor object
        """
        return self * -1

    def __sub__(self, other):
        """
        Returns the output of element-wise subtraction between Tensor objects
        and other non-Tensor object arrays

        Args:
            other: Defines the other operand in the expression
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        """
        Returns the output of a Tensor object divided element-wise by other Tensor objects, 
        and other non-Tensor object arrays


        Args:
            other: Defines the other operand in the expression
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rsub__(self, other):
        """
        Returns the output of element-wise subtraction when operands are reversed

        Args:
            other: Defines the other operand in the expression
        """
        return other + (-self)

    def __radd__(self, other):
        """
        Returns the output of element-wise addition when operands are reversed

        Args:
            other: Defines the other operand in the expression
        """
        return self + other

    def __rmul__(self, other):
        """
        Returns the output of element-wise multiplication when operands are reversed

        Args:
            other: Defines the other operand in the expression
        """
        return self * other

    def __rtruediv__(self, other):
        """
        Returns the output of element-wise division when operands are reversed

        Args:
            other: Defines the other operand in the expression
        """
        return other * (self ** -1)
    
    def matmul(self, other):
        """
        Returns the output of matrix multiplication between Tensor objects
        and non-Tensor objects

        Args:
            other: Defines the other operand in the expression
        """
        out = Tensor((self.data.dot(other.data)),
                     self.requires_grad or other.requires_grad,
                     (self, other),
                     "matmul")
        
        return out

    def sum(self):
        """
        Returns the output of the sum value of all elements in the Tensor object
        """
        out = Tensor(self.data.sum(),
                     self.requires_grad,
                     (self,),
                     "sum")
        
        return out

    def mean(self):
        """
        Returns the output of the mean value of all elements in the Tensor object
        """
        out = Tensor(self.data.mean(),
                     self.requires_grad,
                     (self,),
                     "mean")
        
        return out

    def exp(self):
        """
        Returns the exponential of a Tensor object
        """
        out = Tensor(m.e ** self.data,
                     self.requires_grad,
                     (self,),
                     'exp')

        return out

    def sigmoid(self):
        """
        Returns the sigmoid of a Tensor object
        """
        sigmoid = 1 / (1 + (m.e ** -self.data))
        out = Tensor(sigmoid,
                     self.requires_grad,
                     (self,),
                     'sigmoid')

        return out

    def tanh(self):
        """
        Returns the tanh of a Value object
        """
        tanh = 2*(2*self).sigmoid().data - 1
        out = Value(tanh, (self,), 'tanh')

        return out

    def relu(self):
        pass

    def softmax(self):
        pass

    def backward(self):
        pass

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
