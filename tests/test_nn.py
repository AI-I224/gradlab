"""
This PyTest module tests the methods within the Linear, ReLU and Sequential Class

To run tests, run the following line of code in the terminal:
    'python -m pytest'
"""

import numpy as np
from core.engine import Tensor
from core.nn import Linear, Conv1D, RNNCell, ReLU, Sequential

X = Tensor(np.random.randn(3, 5).astype(np.float32))

def test_linear_forward_shape():
    """
    Tests that the Linear forward pass produces the correct dimension
    """
    layer = Linear(3, 2)
    out = layer(X)
    assert out.data.shape == (2, 5), "Failed producing the correcting dimensions of output Tensor"

def test_conv1d_forward_and_backward():
    """
    Tests that the Conv1D forward pass produces the correct dimension
    and backward updates gradients of weights and biases
    """
    np.random.seed(42)
    batch = 2
    in_channels = 3
    seq_len = 6
    out_channels = 2
    kernel_size = 3

    # Input (in_channels, seq_len, batch)
    x_data = np.random.randn(in_channels, seq_len, batch).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)

    conv = Conv1D(in_channels, out_channels, kernel_size)
    out = conv(x)

    # ---- Forward shape check
    expected_shape = (out_channels, seq_len - kernel_size + 1, batch)
    assert out.data.shape == expected_shape, f"Expected {expected_shape}, got {out.data.shape}"

    # ---- Backward check
    loss = out.sum()
    loss.backward()

    # Check that grads exist and have correct shape
    assert conv.weight.grad.shape == conv.weight.data.shape, "conv.weight shape mismatch"
    assert conv.bias.grad.shape == conv.bias.data.shape, "conv.bias shape mismatch"
    assert x.grad.shape == x.data.shape, "x shape mismatch"

    # Gradients should not be all zeros
    assert not np.allclose(conv.weight.grad, 0), "Weight gradient is zero"
    assert not np.allclose(conv.bias.grad, 0), "Bias gradient is zero"
    assert not np.allclose(x.grad, 0), "Input gradient is zero"

def test_rnncell_forward_and_backward():
    """
    Tests that the RNNCell forward pass produces the correct dimension
    and backward updates gradients of weights and biases
    """
    np.random.seed(42)
    input_size = 4
    hidden_size = 3
    batch = 2

    x_data = np.random.randn(input_size, batch).astype(np.float32)
    h_prev_data = np.random.randn(hidden_size, batch).astype(np.float32)

    x = Tensor(x_data, requires_grad=True)
    h_prev = Tensor(h_prev_data, requires_grad=True)

    rnn = RNNCell(input_size, hidden_size)
    h = rnn(x, h_prev)

    # ---- Forward pass shape check
    assert h.data.shape == (hidden_size, batch)

    # ---- Backward pass check
    loss = h.sum()
    loss.backward()

    # Sanity checks for gradient shapes
    assert rnn.wxh.grad.shape == rnn.wxh.data.shape, "Wxh shape mismatch"
    assert rnn.whh.grad.shape == rnn.whh.data.shape, "Whh shape mismatch"
    assert rnn.bh.grad.shape == rnn.bh.data.shape, "bh shape mismatch"
    assert x.grad.shape == x.data.shape, "x shape mismatch"
    assert h_prev.grad.shape == h_prev.data.shape, "h_prev shape mismatch"

    # Sanity checks for non-zero gradients
    assert not np.allclose(rnn.wxh.grad, 0), "Wxh grad is zero"
    assert not np.allclose(rnn.whh.grad, 0), "Whh grad is zero"
    assert not np.allclose(rnn.bh.grad, 0), "bh grad is zero"
    assert not np.allclose(x.grad, 0), "x grad is zero"
    assert not np.allclose(h_prev.grad, 0), "h_prev grad is zero"


def test_sequential_forward_and_backward():
    """
    Tests that multiple layers in the neural network are chained together
    so that forward pass and backpropagation occur correctly
    """
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2)
    )

    # Forward pass one sample at a time to avoid broadcasting issues
    batch_size = X.data.shape[1]
    out_list = []
    for i in range(batch_size):
        x_i = Tensor(X.data[:, i:i+1], requires_grad=True)
        out_i = model(x_i)
        out_list.append(out_i)

    # Combine outputs manually
    out = Tensor(np.column_stack([o.data for o in out_list]))

    loss = out.sum()
    loss.backward()

    # Check that all parameters have gradients of correct shape
    for p in model.parameters():
        assert p.grad.shape == p.data.shape, "The multiple layers are chained incorrectly"
