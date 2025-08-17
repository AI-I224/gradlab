"""
This Python module handles the optimisation methods used to minimise the loss function,
containing the following modules:

Optimiser, SGD, Adam
"""

import numpy as np

class Optimiser:
    """
    The superclass keeps track of the parameters and provides a class
    for using different optimisation methods
    """
    def __init__(self, params):
        self.params = list(params) # Store all trainable parameters

    def zero_grad(self):
        """
        Reset gradients for all the parameters 
        """
        for p in self.params:
            if p.requires_grad:
                p.zero_grad()

    def step(self):
        """
        Update parameters
        """
        raise NotImplementedError
    