"""
This module contains abstract base classes and concrete implementations
for loss functions used to train neural networks.
"""

import math
from abc import ABC, abstractmethod


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


class BinaryCrossEntropy(LossFunction):
    """Binary Cross Entropy (BCE) loss function."""

    def __call__(self, predicted: list[float], expected: list[float]) -> float:
        assert len(predicted) == len(expected)
        eps = 1e-15
        loss = 0.0
        for p, e in zip(predicted, expected):
            p = max(eps, min(1.0 - eps, p))
            loss -= e * math.log(p) + (1.0 - e) * math.log(1.0 - p)
        return loss / len(predicted)

    def derivative(self, predicted: list[float], expected: list[float]) -> list[float]:
        assert len(predicted) == len(expected)
        eps = 1e-15
        grads = []
        for p, e in zip(predicted, expected):
            p = max(eps, min(1.0 - eps, p))
            grads.append((p - e) / (p * (1.0 - p)))
        return grads


class CategoricalCrossEntropy(LossFunction):
    """Categorical Cross Entropy (CCE) loss function."""

    def __call__(self, predicted: list[float], expected: list[float]) -> float:
        assert len(predicted) == len(expected)
        eps = 1e-15
        loss = 0.0
        for p, e in zip(predicted, expected):
            p = max(eps, min(1.0 - eps, p))
            loss -= e * math.log(p)
        return loss

    def derivative(self, predicted: list[float], expected: list[float]) -> list[float]:
        assert len(predicted) == len(expected)
        eps = 1e-15
        grads = []
        for p, e in zip(predicted, expected):
            p = max(eps, min(1.0 - eps, p))
            grads.append(-e / p)
        return grads
