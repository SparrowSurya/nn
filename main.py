import math
import random
from dataclasses import dataclass, field
from typing import Callable, Self


@dataclass
class NeuralLayer:
    """Represents a single layer in neural network."""

    weights: list[list[float]]
    """Represents the weights in the layer."""

    bias: list[float]
    """Represents the bias in the layer."""

    activation: Callable[[float], float] | None = None
    """Reprsents activation function."""

    @classmethod
    def from_size(cls, input_size: int, output_size: int, activation: Callable[[float], float] | None = None) -> Self:
        """Creates random values from input and output sizes."""
        weights = [[random.random() for _ in range(input_size)] for _ in range(output_size)]
        bias = [0.0 for _ in range(output_size)]
        return cls(weights, bias, activation)

    def __post_init__(self):
        assert(len(self.weights) == len(self.bias))

    @property
    def input_size(self) -> int:
        """Provides input values size for this layer."""
        assert(len(self.weights) > 0)
        return len(self.weights[0])

    @property
    def output_size(self) -> int:
        """Provides input values size for this layer."""
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



def sigmoid(value: float) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1 + math.pow(math.e, -value))


def relu(value: float) -> float:
    """ReLU activation function."""
    return max(0.0, value)


def mean_squared_error(predicted: list[float], expected: list[float]) -> float:
    """Mean square error loss function."""
    assert(len(predicted) == len(expected))
    return sum(math.pow(e-p, 2) for p, e in zip(predicted, expected)) / 2


def binary_cross_entropy(predicted: list[float], expected: list[float], epsilon: float = 1e-15) -> float:
    """Binary cross-entropy loss function."""
    assert(len(predicted) == len(expected))

    total_loss = 0.0
    for p, e in zip(predicted, expected):
        # Clip predicted values between epsilon and 1 - epsilon
        p = max(epsilon, min(1.0 - epsilon, p))

        # Calculate BCE formula: -(y * log(p) + (1 - y) * log(1 - p))
        total_loss += -(e * math.log(p) + (1.0 - e) * math.log(1.0 - p))

    return total_loss / len(predicted)


@dataclass
class NeuralNetwork:
    """Neural network builder class."""

    layers: list[NeuralLayer] = field(default_factory=lambda: [])
    """Sequence of layers."""

    def add_layer(self, layer: NeuralLayer):
        """Adds new layer next to last layer."""
        self.layers.append(layer)

    def validate(self) -> bool:
        """Verifies if the output layer matches to input layer of next layer."""
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

    def train(self):
        pass


def main():
    nn = NeuralNetwork()
    nn.add_layer(NeuralLayer.from_size(2, 3, relu))
    nn.add_layer(NeuralLayer.from_size(3, 1, sigmoid))
    if nn.validate():
        inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        for x in inputs:
            prediction = nn.forward(x)
            print(f"Input: {x} -> Prediction: {prediction}")
    else:
        print("Invalid neural network")


if __name__ == "__main__":
    main()
