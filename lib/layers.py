"""
This module defines the representation of a single neural network layer,
including weight matrix and bias vector.
"""

import random
from dataclasses import dataclass
from typing import Self, Callable
from lib.activations import ActivationFunction


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
