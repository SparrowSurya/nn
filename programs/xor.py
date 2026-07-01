from typing import Callable, Any
import tkinter as tk
from lib.program import NeuralNetworkProgram, BaseTrainingParams
from lib.activations import ReLU, Sigmoid
from lib.losses import MeanSquaredError, LossFunction
from lib.layers import NeuralLayer
from lib.network import NeuralNetwork
from lib.gui import NeuralNetworkGui


class XorTrainingParams(BaseTrainingParams):
    """Specific hyperparameters required to train the XOR model."""
    pass


class XorTaskProgram(NeuralNetworkProgram[list[list[float]], list[list[float]], XorTrainingParams]):
    """Implementation of the XOR logic gate task program."""

    @property
    def name(self) -> str:
        return "XOR Gate"

    def build_network(self) -> NeuralNetwork:
        nn = NeuralNetwork()
        nn.add_layer(NeuralLayer.from_size(2, 3, ReLU()))
        nn.add_layer(NeuralLayer.from_size(3, 1, Sigmoid()))
        return nn

    def _load_data(self) -> tuple[list[list[float]], list[list[float]]]:
        # Return hardcoded XOR training data
        inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        targets = [[0.0], [1.0], [1.0], [0.0]]
        return inputs, targets

    def get_loss_function(self) -> LossFunction:
        return MeanSquaredError()

    def decode_output(self, raw_prediction: list[float]) -> str:
        prob = raw_prediction[0]
        predicted_class = 1 if prob >= 0.5 else 0
        return f"Class: {predicted_class} (Confidence: {prob:.2%})"

    def get_default_training_params(self) -> XorTrainingParams:
        return {"epochs": 10000, "learning_rate": 0.2}


class XorGui(NeuralNetworkGui):
    """GUI extension class for the XOR logic gate task."""

    def build_manual_input_widget(self, parent: Any, on_predict: Callable[[list[float]], None]) -> Any:
        frame = tk.Frame(parent, bg=self.BG_COLOR)
        frame.pack(pady=10)
        
        tk.Label(frame, text="Quick Manual Testing (Click an option):", font=("Arial", 11, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(pady=5)
        
        btn_container = tk.Frame(frame, bg=self.BG_COLOR)
        btn_container.pack(pady=5)
        
        options = [
            ([0.0, 0.0], "00"),
            ([0.0, 1.0], "01"),
            ([1.0, 0.0], "10"),
            ([1.0, 1.0], "11")
        ]
        
        for x, label in options:
            btn = tk.Button(
                btn_container,
                text=label,
                font=("Arial", 11),
                command=lambda val=x: on_predict(val),
                width=5
            )
            btn.pack(side=tk.LEFT, padx=5)
            
        return frame
