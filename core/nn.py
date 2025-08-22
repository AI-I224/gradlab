"""
This Python module defines the structure and chaining of different layers
and activation functions within a neural network, covering the following submodules:

Module, Linear, Exp, Sigmoid, Tanh, ReLU, Sequential
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
        out = self.weight.matmul(x) + self.bias

        def _backward():
            if self.weight.requires_grad:
                # dL/dW = dL/dout @ x^T
                self.weight.grad += out.grad @ x.data.T
            if x.requires_grad:
                # dL/dx = W^T @ dL/dout
                x.grad += self.weight.data.T @ out.grad
            if self.bias.requires_grad:
                # dL/db = sum over batch
                self.bias.grad += np.sum(out.grad, axis=1, keepdims=True)

        out._backward = _backward
        out._prev = {x, self.weight, self.bias}
        return out

    def parameters(self):
        """
        Returns the weights and biases for each parameter in the layer
        """
        return [self.weight, self.bias]

class Conv1D(Module):
    """
    A simple 1D convolution layer for sequences

    Performs a convolution of the input over the sequence dimension
    with multiple output channels

    Attributes:
        in_channels: number of input channels/features
        out_channels: number of output channels/features
        kernel_size: width of the convolutional kernel
        requires_grad: A boolean that decides whether the autograd engine 
          tracks the Tensor's operations
        weight: Tensor of shape representing convolutional kernels
        bias: Tensor of shape added to each output channel
    """
    def __init__(self, in_channels, out_channels, kernel_size, requires_grad=True):
        self.kernel_size = kernel_size
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32) * 0.01,
            requires_grad=requires_grad
        )
        self.bias = Tensor(
            np.zeros((out_channels, 1), dtype=np.float32),
            requires_grad=requires_grad
        )

    def __call__(self, x: Tensor) -> Tensor:
        _, seq_len, batch = x.data.shape  # _ == in_channels
        out_channels, _, k = self.weight.data.shape # _ == in_channels
        new_len = seq_len - k + 1

        out_data = np.zeros((out_channels, new_len, batch), dtype=np.float32)
        for b in range(batch):
            for oc in range(out_channels):
                for i in range(new_len):
                    region = x.data[:, i:i+k, b]
                    w = self.weight.data[oc]
                    out_data[oc, i, b] = np.sum(region * w) + self.bias.data[oc, 0]

        out = Tensor(out_data, requires_grad=x.requires_grad or self.weight.requires_grad)

        def _backward():
            """
            Computes gradient from backward pass
            """
            if self.weight.requires_grad:
                for b in range(batch):
                    for oc in range(out_channels):
                        for i in range(new_len):
                            grad_out = out.grad[oc, i, b]
                            region = x.data[:, i:i+k, b]
                            self.weight.grad[oc] += grad_out * region
            if x.requires_grad:
                for b in range(batch):
                    for oc in range(out_channels):
                        for i in range(new_len):
                            grad_out = out.grad[oc, i, b]
                            x.grad[:, i:i+k, b] += grad_out * self.weight.data[oc]
            if self.bias.requires_grad:
                for oc in range(out_channels):
                    self.bias.grad[oc] += np.sum(out.grad[oc])

        out._backward = _backward
        out._prev = {x, self.weight, self.bias}
        return out

    def parameters(self):
        """
        Returns the weights and biases for each parameter in the layer
        """
        return [self.weight, self.bias]

class RNNCell(Module):
    """
    A simple RNN cell.

    Performs a single timestep update of hidden state:
        h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)

    Args:
        input_size: number of features in input x_t.
        hidden_size: number of hidden units.
        requires_grad: A boolean that decides whether the autograd engine 
          tracks the Tensor's operations
        wxh: weight matrix for input-to-hidden connections.
        whh: weight matrix for hidden-to-hidden connections.
        bh: bias vector for hidden state update.
    """
    def __init__(self, input_size, hidden_size, requires_grad=True):
        self.hidden_size = hidden_size
        self.wxh = Tensor(
            np.random.randn(hidden_size, input_size).astype(np.float32) * 0.01,
            requires_grad=requires_grad
        )
        self.whh = Tensor(
            np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.01,
            requires_grad=requires_grad
        )
        self.bh = Tensor(
            np.zeros((hidden_size, 1), dtype=np.float32),
            requires_grad=requires_grad
        )

    def __call__(self, x: Tensor, h_prev: Tensor) -> Tensor:
        z = self.wxh.matmul(x) + self.whh.matmul(h_prev) + self.bh
        h = z.tanh()

        def _backward():
            """
            Computes gradient from backward pass
            """
            dz = (1 - h.data**2) * h.grad  # derivative of tanh
            if self.wxh.requires_grad:
                self.wxh.grad += dz @ x.data.T
            if self.whh.requires_grad:
                self.whh.grad += dz @ h_prev.data.T
            if self.bh.requires_grad:
                self.bh.grad += np.sum(dz, axis=1, keepdims=True)
            if x.requires_grad:
                x.grad += self.wxh.data.T @ dz
            if h_prev.requires_grad:
                h_prev.grad += self.whh.data.T @ dz

        h._backward = _backward
        h._prev = {x, h_prev, self.wxh, self.whh, self.bh}
        return h

    def parameters(self):
        return [self.wxh, self.whh, self.bh]

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
    