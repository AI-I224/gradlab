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
    def __init__(self):
        pass

class ReLU:
    def __init__(self):
        pass

class Sequential:
    def __init__(self):
        pass