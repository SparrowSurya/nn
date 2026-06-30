"""
This module defines the abstract blueprint for running different
neural network tasks/programs, using Python 3.12+ PEP 695 type parameter syntax
with bounds to ensure type safety for input/output dataset structures.
It also implements the RunObserver pattern to modularize run reporting.
"""

from abc import ABC, abstractmethod
from lib.network import NeuralNetwork
from lib.losses import LossFunction


class RunObserver(ABC):
    """Abstract base class representing an observer for the entire program execution pipeline."""

    @abstractmethod
    def on_run_start(self, program_name: str):
        """Called when the program execution starts."""
        pass

    @abstractmethod
    def on_predictions_start(self, phase: str):
        """Called before starting the list of prediction outputs (e.g., 'BEFORE', 'AFTER')."""
        pass

    @abstractmethod
    def on_prediction(self, input_val: list[float], decoded_prediction: str):
        """Called for each individual input prediction during evaluation."""
        pass

    @abstractmethod
    def on_training_start(self):
        """Called when the training process begins."""
        pass

    @abstractmethod
    def on_validation_error(self, message: str):
        """Called when neural network configuration validation fails."""
        pass


class ConsoleRunObserver(RunObserver):
    """Default implementation of RunObserver that prints step progression to the console."""

    def on_run_start(self, program_name: str):
        print(f"Executing program: {program_name}")

    def on_predictions_start(self, phase: str):
        print(f"\n--- Predictions {phase} training ---")

    def on_prediction(self, input_val: list[float], decoded_prediction: str):
        print(f"Input: {input_val} -> Prediction: {decoded_prediction}")

    def on_training_start(self):
        print("\n--- Training ---")

    def on_validation_error(self, message: str):
        print(message)


# Constrain T_In and T_Out to be list[list[float]] to ensure neural network compatibility
class NeuralNetworkProgram[T_In: list[list[float]], T_Out: list[list[float]], T_Params](ABC):
    """Abstract base class representing a single neural network task/program."""

    def __init__(self):
        self.inputs: T_In | None = None
        self.targets: T_Out | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the user-friendly name of the task (e.g., 'XOR Gate')."""
        pass

    @abstractmethod
    def build_network(self) -> NeuralNetwork:
        """Initializes and returns the specific NeuralNetwork layer structure for this task."""
        pass

    def load_training_data(self) -> tuple[T_In, T_Out]:
        """Loads and caches the training dataset, returning (inputs, targets)."""
        if self.inputs is None or self.targets is None:
            self.inputs, self.targets = self._load_data()
        return self.inputs, self.targets

    @abstractmethod
    def _load_data(self) -> tuple[T_In, T_Out]:
        """Loads and prepares the raw training dataset from its source."""
        pass

    @abstractmethod
    def get_loss_function(self) -> LossFunction:
        """Returns the appropriate LossFunction."""
        pass

    @abstractmethod
    def decode_output(self, raw_prediction: list[float]) -> str:
        """Translates raw model predictions into a user-friendly string."""
        pass

    @abstractmethod
    def get_default_training_params(self) -> T_Params:
        """Returns the default training hyperparameters."""
        pass

    def run(
        self,
        observer: RunObserver | None = None,
        show_loss: bool = False,
        show_network: bool = False,
    ):
        """Orchestrates the entire data loading, network validation, training, and visualization pipeline."""
        if observer is None:
            observer = ConsoleRunObserver()

        observer.on_run_start(self.name)

        # 1. Build and validate network
        nn = self.build_network()
        if not nn.validate():
            observer.on_validation_error("Invalid neural network configuration.")
            return

        # 2. Load training data (uses caching internally)
        inputs, targets = self.load_training_data()

        observer.on_predictions_start("BEFORE")
        for x in inputs:
            prediction = nn.forward(x)
            observer.on_prediction(x, self.decode_output(prediction))

        observer.on_training_start()

        # Setup modular epoch observers
        from lib.observers import ConsoleEpochObserver, PlotEpochObserver, CompositeEpochObserver
        console_observer = ConsoleEpochObserver(frequency=2000)
        plot_observer = PlotEpochObserver()
        composite_observer = CompositeEpochObserver([console_observer, plot_observer])

        # 3. Train the model using task parameters
        params = self.get_default_training_params()

        # Handle params if it is a dictionary (like TypedDict) or a custom object
        if isinstance(params, dict):
            epochs = params["epochs"]
            learning_rate = params["learning_rate"]
        else:
            epochs = getattr(params, "epochs")
            learning_rate = getattr(params, "learning_rate")

        nn.train(
            inputs,
            targets,
            epochs=epochs,
            learning_rate=learning_rate,
            loss_fn=self.get_loss_function(),
            observer=composite_observer,
        )

        observer.on_predictions_start("AFTER")
        for x in inputs:
            prediction = nn.forward(x)
            observer.on_prediction(x, self.decode_output(prediction))

        # Plot visual metrics if requested
        if show_loss:
            plot_observer.plot()
        if show_network:
            from lib.visualizations import visualize_network_structure
            visualize_network_structure(nn)
