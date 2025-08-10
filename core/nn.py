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

class Linear:
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
            np.random.randn(nin, nout).astype(np.float32) * 0.01,
            requires_grad=requires_grad
        )
        self.bias = Tensor(
            np.zeros((nout, 1), dtype=np.float32),
            requires_grad=requires_grad
        )

class ReLU:
    def __init__(self):
        pass

class Sequential:
    def __init__(self):
        pass