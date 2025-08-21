"""
This Python module acts as the training loop manager for GradLab, handling the training
and evaluation methods for the models built. Also provides the loss functions.
The module contains the following modules:

Trainer, MSELoss, CrossEntropyLoss
"""
import numpy as np
from core.engine import Tensor

class Trainer:
    """
    A simple training loop manager for neural network models.

    Attribues:
        model: A neural network
        optimizer: An optimizer responsible for parameter updates
        loss_fn: A loss function used to measure prediction error
    """
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, x_batch, y_batch):
        """
        Perform a single training step on one batch of data and
        returns the scalar loss value for this batch

        Args:
            x_batch: input data of shape (features, batch_size)
            y_batch: target labels of shape (output_dim, batch_size)
        """
        preds = self.model(x_batch)
        loss = self.loss_fn(preds, y_batch)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.data.item() if hasattr(loss.data, "item") else float(loss.data)

    def train(self, X, Y, epochs=10, batch_size=None):
        """
        Train the model for a given number of epochs on the dataset

        Args:
            X: input features of shape
            Y: target labels of shape 
            epochs: number of full passes through the dataset
            batch_size: mini-batch size
        """
        n = X.shape[1] 
        for epoch in range(epochs):
            if batch_size is None:
                batch_size = n  # full batch

            # Shuffle dataset
            indices = np.random.permutation(n)
            X, Y = X[:, indices], Y[:, indices]

            losses = []
            for start in range(0, n, batch_size):
                end = start + batch_size
                x_batch = Tensor(X[:, start:end], requires_grad=True)
                y_batch = Tensor(Y[:, start:end])

                loss = self.train_step(x_batch, y_batch)
                losses.append(loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")
