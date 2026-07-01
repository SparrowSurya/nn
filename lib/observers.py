"""
This module implements the observer pattern to monitor, log, and visualize
training progress and execution steps of the neural network.
"""

from abc import ABC


class RunObserver(ABC):
    """Base class representing an observer for the entire program execution pipeline.
    
    Subclasses can override only the specific methods they are interested in.
    By default, all callback hooks are no-ops.
    """

    def on_run_start(self, program_name: str):
        """Called when the program execution starts."""
        pass

    def on_predictions_start(self, phase: str):
        """Called before starting the list of prediction outputs (e.g., 'BEFORE', 'AFTER')."""
        pass

    def on_prediction(self, input_repr: str, target_repr: str, prediction_repr: str):
        """Called for each individual input prediction during evaluation."""
        pass

    def on_training_start(self):
        """Called when the training process begins."""
        pass

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        """Called at the end of each training epoch."""
        pass

    def on_validation_error(self, message: str):
        """Called when neural network configuration validation fails."""
        pass


class ConsoleRunObserver(RunObserver):
    """Epoch observer that logs execution steps and training progress to the console."""

    frequency: int
    """How often (in epochs) progress should be logged to the console."""

    def __init__(self, frequency: int = 1000):
        self.frequency = frequency

    def on_run_start(self, program_name: str):
        print(f"Executing program: {program_name}")

    def on_predictions_start(self, phase: str):
        print(f"\n--- Predictions {phase} training ---")

    def on_prediction(self, input_repr: str, target_repr: str, prediction_repr: str):
        print(f"Input: {input_repr} -> Expected: {target_repr} | Predicted: {prediction_repr}")

    def on_training_start(self):
        print("\n--- Training ---")

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        if epoch % self.frequency == 0:
            print(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.6f}")

    def on_validation_error(self, message: str):
        print(message)


class PlotRunObserver(RunObserver):
    """Observer that records loss at each epoch and plots it using matplotlib."""

    losses: list[float]
    """Records loss at each epoch."""

    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
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


class CompositeRunObserver(RunObserver):
    """Observer that delegates callbacks to multiple other observers."""

    observers: list[RunObserver]
    """List of child observers to delegate callbacks to."""

    def __init__(self, observers: list[RunObserver]):
        self.observers = observers

    def on_run_start(self, program_name: str):
        for observer in self.observers:
            observer.on_run_start(program_name)

    def on_predictions_start(self, phase: str):
        for observer in self.observers:
            observer.on_predictions_start(phase)

    def on_prediction(self, input_repr: str, target_repr: str, prediction_repr: str):
        for observer in self.observers:
            observer.on_prediction(input_repr, target_repr, prediction_repr)

    def on_training_start(self):
        for observer in self.observers:
            observer.on_training_start()

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
        for observer in self.observers:
            observer.on_epoch_end(epoch, total_epochs, loss)

    def on_validation_error(self, message: str):
        for observer in self.observers:
            observer.on_validation_error(message)
