import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self, Callable


class ActivationFunction(ABC):
    """Abstract base class representing an activation function and its derivative."""

    @abstractmethod
    def __call__(self, value: float) -> float:
        """Computes the activation function for a given input value."""
        pass

    @abstractmethod
    def derivative(self, activated_value: float) -> float:
        """Computes the derivative of the activation function given its activated output."""
        pass


class Sigmoid(ActivationFunction):
    """Sigmoid activation function."""

    def __call__(self, value: float) -> float:
        return 1.0 / (1 + math.pow(math.e, -value))

    def derivative(self, activated_value: float) -> float:
        return activated_value * (1.0 - activated_value)


class ReLU(ActivationFunction):
    """Rectified Linear Unit (ReLU) activation function."""

    def __call__(self, value: float) -> float:
        return max(0.0, value)

    def derivative(self, activated_value: float) -> float:
        return 1.0 if activated_value > 0.0 else 0.0


class LossFunction(ABC):
    """Abstract base class representing a loss function and its derivative."""

    @abstractmethod
    def __call__(self, predicted: list[float], expected: list[float]) -> float:
        """Computes the loss value given predicted and expected lists."""
        pass

    @abstractmethod
    def derivative(self, predicted: list[float], expected: list[float]) -> list[float]:
        """Computes the gradient (derivatives) of the loss with respect to predictions."""
        pass


class MeanSquaredError(LossFunction):
    """Mean Squared Error (MSE) loss function."""

    def __call__(self, predicted: list[float], expected: list[float]) -> float:
        assert len(predicted) == len(expected)
        return sum(math.pow(e - p, 2) for p, e in zip(predicted, expected)) / 2

    def derivative(self, predicted: list[float], expected: list[float]) -> list[float]:
        assert len(predicted) == len(expected)
        return [p - e for p, e in zip(predicted, expected)]


class EpochObserver(ABC):
    """Abstract base class representing an observer for training epochs."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        """Called at the end of each epoch with the current status."""
        pass


class ConsoleEpochObserver(EpochObserver):
    """Epoch observer that logs progress to the console at a set frequency."""

    frequency: int
    """How often (in epochs) progress should be logged to the console."""

    def __init__(self, frequency: int = 1000):
        self.frequency = frequency

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        """Callback to log output of current epoch to console."""
        if epoch % self.frequency == 0:
            print(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.6f}")


@dataclass
class NeuralLayer:
    """Represents a single layer in neural network."""

    weights: list[list[float]]
    """Represents the weights in the layer."""

    bias: list[float]
    """Represents the bias in the layer."""

    activation: ActivationFunction | None = None
    """Represents the activation function."""

    @classmethod
    def from_size(
        cls,
        input_size: int,
        output_size: int,
        activation: ActivationFunction | None = None,
        random_func: Callable[[], float] | None = None,
    ) -> Self:
        """Creates random values from input and output sizes."""
        random_func = random.random if random_func is None else random_func
        weights = [[random_func() for _ in range(input_size)] for _ in range(output_size)]
        bias = [random_func() for _ in range(output_size)]
        return cls(weights, bias, activation)

    def __post_init__(self):
        assert len(self.weights) == len(self.bias)

    @property
    def input_size(self) -> int:
        """Provides input values size for this layer."""
        assert len(self.weights) > 0
        return len(self.weights[0])

    @property
    def output_size(self) -> int:
        """Provides output values size for this layer."""
        return len(self.bias)

    def forward(self, layer_input: list[float]) -> list[float]:
        """Computes the forward pass for this layer."""
        layer_output = []

        # Each neuron has its own list of weights and a bias
        for neuron_weights, neuron_bias in zip(self.weights, self.bias):
            # Calculate dot product: sum(input_i * weight_i) + bias
            z = sum(x * w for x, w in zip(layer_input, neuron_weights)) + neuron_bias

            # Apply the activation function if one is defined
            if self.activation:
                z = self.activation(z)

            layer_output.append(z)

        return layer_output


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

    def train(
        self,
        inputs: list[list[float]],
        targets: list[list[float]],
        epochs: int,
        learning_rate: float,
        loss_fn: LossFunction,
        observer: EpochObserver | None = None,
    ):
        """Trains the network using backpropagation and gradient descent."""
        assert len(inputs) == len(targets)

        for epoch in range(epochs):
            total_loss = 0.0

            for x, y_true in zip(inputs, targets):
                # --- 1. FORWARD PASS (with caching) ---
                layer_inputs = []
                current_activation = x

                for layer in self.layers:
                    layer_inputs.append(current_activation)
                    current_activation = layer.forward(current_activation)

                y_pred = current_activation
                total_loss += loss_fn(y_pred, y_true)

                # --- 2. BACKWARD PASS (Backpropagation) ---
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

            # Notify observer at the end of each epoch
            if observer is not None:
                observer.on_epoch_end(epoch, epochs, total_loss / len(inputs))


def main():
    random.seed(42)

    nn = NeuralNetwork()
    nn.add_layer(NeuralLayer.from_size(2, 3, ReLU(), random_func=lambda: random.uniform(-1.0, 1.0)))
    nn.add_layer(NeuralLayer.from_size(3, 1, Sigmoid(), random_func=lambda: random.uniform(-1.0, 1.0)))

    if not nn.validate():
        print("Invalid neural network configuration.")
        return

    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]

    print("--- Predictions BEFORE training ---")
    for x in inputs:
        prediction = nn.forward(x)
        print(f"Input: {x} -> Prediction: {prediction}")

    print("\n--- Training ---")
    observer = ConsoleEpochObserver(frequency=2000)
    nn.train(
        inputs,
        targets,
        epochs=10000,
        learning_rate=0.2,
        loss_fn=MeanSquaredError(),
        observer=observer,
    )

    print("\n--- Predictions AFTER training ---")
    for x in inputs:
        prediction = nn.forward(x)
        print(f"Input: {x} -> Prediction: {prediction}")


if __name__ == "__main__":
    main()
