"""
This Python module handles the optimisation methods used to minimise the loss function,
containing the following modules:

Optimiser, SGD, Adam
"""

import numpy as np

class Optimiser:
    """
    The superclass keeps track of the parameters and provides a structure
    for implementing different optimisation methods
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
        Base method for update parameters in a single optimisation step,
        which should be implemented by subclasses (i.e. SGD, Adam)
        """
        raise NotImplementedError

class SGD(Optimiser):
    """
    Implements Stochastic Gradient Descent Optimisation Algorithm
    """
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
    
    def step(self):
        for p, v in zip(self.params, self.velocities):
            if p.requires_grad:
                if self.momentum:
                    v[:] = self.momentum * v - self.lr * p.grad
                    p.data += v
                else:
                    p.data -= self.lr * p.grad

class AdaGrad(Optimiser):
    def __init__(self, params):
        super().__init__(params)

class RMSProp(Optimiser):
    def __init__(self, params):
        super().__init__(params)

class Adam(Optimiser):
    """
    Implements Adam Optimisation Algorithm
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
        self.v = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas

        for p, m, v in zip(self.params, self.m, self.v):
            if p.requires_grad:
                m[:] = b1 * m + (1 - b1) * p.grad
                v[:] = b2 * v + (1 - b2) * (p.grad ** 2)

                m_hat = m / (1 - b1 ** self.t)
                v_hat = v / (1 - b2 ** self.t)

                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
