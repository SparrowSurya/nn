"""
This module implements the observer pattern to monitor, log, and visualize
training progress at the end of each epoch.
"""

from abc import ABC, abstractmethod


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


class PlotEpochObserver(EpochObserver):
    """Epoch observer that records loss at each epoch and plots it using matplotlib."""

    losses: list[float]
    """Records loss at each epoch."""

    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        """Records loss value of current epoch."""
        self.losses.append(loss)

    def plot(self):
        """Plots the loss history using matplotlib."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, label="Training Loss", color="royalblue", linewidth=2)
        plt.title("Loss Progression over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.show()


class CompositeEpochObserver(EpochObserver):
    """Observer that delegates callbacks to multiple other epoch observers."""

    def __init__(self, observers: list[EpochObserver]):
        self.observers = observers

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        """Passes the callback execution to all registered observers."""
        for observer in self.observers:
            observer.on_epoch_end(epoch, total_epochs, loss)
