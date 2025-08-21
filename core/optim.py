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

    Attributes:
        params: list of all trainable parameters
    """
    def __init__(self, params):
        self.params = list(params)

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

    Attributes:
        params: list of all trainable parameters
        lr: controls how much to change the model in response to the estimated error
          each time the model weights are updated
        momentum: adds a fraction of the previous update to the current one,
          simulating inertia which helps smooth oscillations and speed up
          convergence
    """
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
    
    def step(self):
        """
        Optimisation step for algorithm
        """
        for p, v in zip(self.params, self.velocities):
            if p.requires_grad:
                if self.momentum:
                    v[:] = self.momentum * v - self.lr * p.grad
                    p.data += v
                else:
                    p.data -= self.lr * p.grad

class AdaGrad(Optimiser):
    """
    Implements AdaGrad Optimisation Algorithm

    Attributes:
        params: list of all trainable parameters
        lr: controls how much to change the model in response to the estimated error
          each time the model weights are updated
        eps: a constant to prevent division by zero
    """
    def __init__(self, params, lr=0.01, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.g_sum = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        """
        Optimisation step for algorithm
        """
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            # g_sum is the sum of the squared gradients
            self.g_sum[i] += p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(self.g_sum[i]) + self.eps)

class RMSProp(Optimiser):
    """
    Implements Root Mean Square Propogation Optimisation Algorithm

    Attributes:
        params: list of all trainable parameters
        lr: controls how much to change the model in response to the estimated error
          each time the model weights are updated
        betas: controls how quickly the moving average of squared gradients changes
        eps: a constant to prevent division by zero
    """
    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        """
        Optimisation step for algorithm
        """
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            # v is the exponentially weighted average of the squared gradient at time step, t
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (p.grad ** 2)
            p.data -= self.lr * p.grad / (np.sqrt(self.v[i]) + self.eps)

class Adam(Optimiser):
    """
    Implements Adam Optimisation Algorithm

    Attributes:
        params: list of all trainable parameters
        lr: controls how much to change the model in response to the estimated error
          each time the model weights are updated
        betas: controls how quickly the moving average of squared gradients changes
        eps: a constant to prevent division by zero
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
        """
        Optimisation step for algorithm
        """
        self.t += 1
        b1, b2 = self.betas

        for p, m, v in zip(self.params, self.m, self.v):
            if p.requires_grad:
                m[:] = b1 * m + (1 - b1) * p.grad
                v[:] = b2 * v + (1 - b2) * (p.grad ** 2)

                m_hat = m / (1 - b1 ** self.t)
                v_hat = v / (1 - b2 ** self.t)

                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
