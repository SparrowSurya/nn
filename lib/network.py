"""
This module defines the high-level NeuralNetwork builder class, which orchestrates
forward propagation, backpropagation, and parameters optimization over epochs.
"""

from dataclasses import dataclass, field
from lib.layers import NeuralLayer
from lib.losses import LossFunction
from lib.observers import RunObserver


@dataclass
class NeuralNetwork:
    """Neural network builder class."""

    layers: list[NeuralLayer] = field(default_factory=lambda: [])
    """Sequence of layers."""

    def add_layer(self, layer: NeuralLayer):
        """Adds new layer next to last layer."""
        self.layers.append(layer)

    def validate(self) -> bool:
        """Verifies if the output layer matches the input layer of the next layer."""
        if len(self.layers) == 0:
            return False

        last = self.layers[0]
        for layer in self.layers[1:]:
            if layer.input_size != last.output_size:
                return False
            last = layer

        return True

    def forward(self, input: list[float]) -> list[float]:
        """Performs forward propagation."""
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def _forward_step(self, x: list[float]) -> tuple[list[float], list[list[float]]]:
        """Performs a forward pass and returns (prediction, cached_layer_inputs)."""
        layer_inputs = []
        current_activation = x

        for layer in self.layers:
            layer_inputs.append(current_activation)
            current_activation = layer.forward(current_activation)

        return current_activation, layer_inputs

    def _backward_step(
        self,
        layer_inputs: list[list[float]],
        y_pred: list[float],
        y_true: list[float],
        learning_rate: float,
        loss_fn: LossFunction,
    ):
        """Performs a backward pass (backpropagation) and updates parameters."""
        # Compute gradient of loss with respect to prediction
        layer_gradients = loss_fn.derivative(y_pred, y_true)

        # Iterate backwards from output layer to input layer
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            x_in = layer_inputs[layer_idx]

            next_layer_gradients = [0.0] * len(x_in)
            layer_output = layer.forward(x_in)

            for j in range(layer.output_size):
                output_j = layer_output[j]

                # Calculate delta
                if layer.activation:
                    delta = layer_gradients[j] * layer.activation.derivative(output_j)
                else:
                    delta = layer_gradients[j]

                # Update bias
                layer.bias[j] -= learning_rate * delta

                # Update weights and accumulate next layer gradients
                for i in range(layer.input_size):
                    next_layer_gradients[i] += delta * layer.weights[j][i]
                    layer.weights[j][i] -= learning_rate * delta * x_in[i]

            layer_gradients = next_layer_gradients

    def _train_step(
        self,
        x: list[float],
        y_true: list[float],
        learning_rate: float,
        loss_fn: LossFunction,
    ) -> float:
        """Runs a single forward and backward step for one training sample, returning the loss."""
        # 1. Forward Pass
        y_pred, layer_inputs = self._forward_step(x)

        # 2. Compute Loss
        loss = loss_fn(y_pred, y_true)

        # 3. Backward Pass
        self._backward_step(layer_inputs, y_pred, y_true, learning_rate, loss_fn)

        return loss

    def train(
        self,
        inputs: list[list[float]],
        targets: list[list[float]],
        epochs: int,
        learning_rate: float,
        loss_fn: LossFunction,
        observer: RunObserver | None = None,
    ):
        """Trains the network by running train steps over multiple epochs."""
        assert len(inputs) == len(targets)

        for epoch in range(epochs):
            total_loss = 0.0

            for x, y_true in zip(inputs, targets):
                total_loss += self._train_step(x, y_true, learning_rate, loss_fn)

            # Notify observer at the end of each epoch
            if observer is not None:
                observer.on_epoch_end(epoch, epochs, total_loss / len(inputs))
