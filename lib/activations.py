"""
This module contains abstract base classes and concrete implementations
for activation functions used in neural network layers.
"""

import math
from abc import ABC, abstractmethod


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


class LeakyReLU(ActivationFunction):
    """Leaky Rectified Linear Unit (LeakyReLU) activation function."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, value: float) -> float:
        return value if value > 0.0 else self.alpha * value

    def derivative(self, activated_value: float) -> float:
        return 1.0 if activated_value > 0.0 else self.alpha
