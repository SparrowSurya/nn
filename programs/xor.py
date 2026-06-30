from lib.program import NeuralNetworkProgram, BaseTrainingParams
from lib.activations import ReLU, Sigmoid
from lib.losses import MeanSquaredError, LossFunction
from lib.layers import NeuralLayer
from lib.network import NeuralNetwork


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
