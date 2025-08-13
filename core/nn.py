"""
This Python module defines the following classes:

Module: describes a scalar value and how it interacts with other Value objects
and non-Value numeric objects (ie integer and float)

Linear:

Exp:

Sigmoid:

Tanh:

ReLU:

Sequential:
"""

import numpy as np
from core.engine import Tensor

class Module:
    """
    Base class for neural network components
    """
    def parameters(self):
        """
        Returns all the trainable parameters in this module for the neural network
        """
        return []
    
    def zero_grad(self):
        """
        Reset gradients for all the parameters 
        """
        for p in self.parameters():
            p.zero_grad()

class Linear(Module):
    """
    Defines a layer within the neural network, performaing a linear transformation of the form:

    y = W * x + b

    Attributes:
        nin: size of input Tensor
        nout: size of output Tensor
        requires_grad: A boolean that decides whether the autograd engine 
          tracks the Tensors within the layer
    """
    def __init__(self, nin, nout, requires_grad=True):
        self.weight = Tensor(
            np.random.randn(nout, nin).astype(np.float32) * 0.01,
            requires_grad=requires_grad
        )
        self.bias = Tensor(
            np.zeros((nout, 1), dtype=np.float32),
            requires_grad=requires_grad
        )

    def __call__(self, x: Tensor) -> Tensor:
        """
        Makes the Linear object callable
        """
        # Shape check — may implement broadcasting later
        assert x.data.shape[0] == self.weight.data.shape[1], (
        f"Shape mismatch: expected input with {self.weight.data.shape[1]} features, "
        f"got {x.data.shape[0]}"
        )

        out = self.weight.matmul(x)

        # Expand bias across batch dimension
        batch_size = x.data.shape[1]
        bias_expanded = Tensor(
            np.repeat(self.bias.data, batch_size, axis=1),
            requires_grad=self.bias.requires_grad
        )

        return out + bias_expanded # Add bias explicitly without relying on broadcasting

    def parameters(self):
        """
        Returns the weights and biases for each parameter in the layer
        """
        return [self.weight, self.bias]

class Exp(Module):
    """
    Applies the exponential function to the output values from the previous layer
    """
    def __call__(self, x: Tensor) -> Tensor:
        return x.exp()

class Sigmoid(Module):
    """
    Applies the Sigmoid activation to the output values from the previous layer
    """
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()
    
class Tanh(Module):
    """
    Applies the Tanh activation to the output values from the previous layer
    """
    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()

class ReLU(Module):
    """
    Applies the ReLU activation to the output values from the previous layer
    """
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

class Sequential(Module):
    """
    Builder class for connecting the layers and activation functions
    for the neural network
    """
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """
        Returns all the parameters within the neural network as a single list
        """
        return [p for layer in self.layers for p in layer.parameters()]
    