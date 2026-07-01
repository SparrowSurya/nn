"""
This module defines the abstract blueprint for running different
neural network tasks/programs, using Python 3.12+ PEP 695 type parameter syntax
with bounds to ensure type safety for input/output dataset structures.
It also implements the RunObserver pattern to modularize run reporting.
"""


from abc import ABC, abstractmethod
from typing import TypedDict, NotRequired
from lib.network import NeuralNetwork
from lib.losses import LossFunction
from lib.observers import RunObserver


class BaseTrainingParams(TypedDict):
    """Base TypedDict representing the minimum set of hyperparameters required to train a network."""
    epochs: int
    learning_rate: float
    console_frequency: NotRequired[int]


# Constrain T_In and T_Out to be list[list[float]], and T_Params to BaseTrainingParams to guarantee standard hyperparameters
class NeuralNetworkProgram[T_In: list[list[float]], T_Out: list[list[float]], T_Params: BaseTrainingParams](ABC):
    """Abstract base class representing a single neural network task/program."""

    def __init__(self):
        self.inputs: T_In | None = None
        self.targets: T_Out | None = None

    def repr_input(self, input_val: list[float]) -> str:
        """Returns a string representation of an input vector for logging."""
        if len(input_val) > 10:
            return f"<Vector of size {len(input_val)}>"
        return str(input_val)

    def decode_target(self, raw_target: list[float]) -> str:
        """Translates raw target outputs into a user-friendly string."""
        if len(raw_target) > 1:
            try:
                return str(raw_target.index(max(raw_target)))
            except ValueError:
                pass
        if len(raw_target) == 1:
            val = raw_target[0]
            return f"{int(val)}" if val.is_integer() else f"{val:.4f}"
        return str(raw_target)

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
        observer: RunObserver,
        show_network: bool = False,
        manual_testing: bool = False,
    ):
        """Orchestrates the entire data loading, network validation, training, and visualization pipeline."""
        observer.on_run_start(self.name)

        # 1. Build and validate network
        nn = self.build_network()
        if not nn.validate():
            observer.on_validation_error("Invalid neural network configuration.")
            return

        # 2. Load training data (uses caching internally)
        inputs, targets = self.load_training_data()

        observer.on_predictions_start("BEFORE")
        for x, y in zip(inputs, targets):
            prediction = nn.forward(x)
            observer.on_prediction(self.repr_input(x), self.decode_target(y), self.decode_output(prediction))

        observer.on_training_start()

        # 3. Train the model using task parameters
        params = self.get_default_training_params()
        epochs = params["epochs"]
        learning_rate = params["learning_rate"]

        nn.train(
            inputs,
            targets,
            epochs=epochs,
            learning_rate=learning_rate,
            loss_fn=self.get_loss_function(),
            observer=observer,
        )

        observer.on_predictions_start("AFTER")
        for x, y in zip(inputs, targets):
            prediction = nn.forward(x)
            observer.on_prediction(self.repr_input(x), self.decode_target(y), self.decode_output(prediction))

        # Plot visual metrics if requested
        if show_network:
            from lib.visualizations import visualize_network_structure
            visualize_network_structure(nn)

        if manual_testing:
            observer.on_test_model(nn)
