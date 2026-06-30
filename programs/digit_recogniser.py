import struct
import os
from lib.program import NeuralNetworkProgram, BaseTrainingParams
from lib.activations import ReLU, Sigmoid
from lib.losses import MeanSquaredError, LossFunction
from lib.layers import NeuralLayer
from lib.network import NeuralNetwork


class DigitRecogniserTrainingParams(BaseTrainingParams):
    """Specific hyperparameters required to train the digit recogniser model."""
    pass


class DigitRecogniserTaskProgram(NeuralNetworkProgram[list[list[float]], list[list[float]], DigitRecogniserTrainingParams]):
    """Implementation of the digit recogniser task program."""

    @property
    def name(self) -> str:
        return "Digit recogniser"

    def build_network(self) -> NeuralNetwork:
        # An MNIST image has 28x28 pixels = 784 inputs.
        # We classify into 10 possible outputs (digits 0 to 9).
        # We use a single hidden layer of size 32. This hidden size is selected as a
        # balanced compromise: wide enough to learn digits, but small enough to run
        # quickly in pure Python without hardware acceleration.
        nn = NeuralNetwork()
        nn.add_layer(NeuralLayer.from_size(784, 32, ReLU()))
        nn.add_layer(NeuralLayer.from_size(32, 10, Sigmoid()))
        return nn

    def _load_data(self) -> tuple[list[list[float]], list[list[float]]]:
        # Pure Python list-based neural networks do not benefit from SIMD/GPU acceleration.
        # Running all 60,000 training images would take a very long time in vanilla Python.
        # Therefore, we limit loading to the first 1,000 images by default.
        # You can increase this limit to train a more accurate model when needed.
        limit = 1000

        # Define file paths dynamically
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        images_path = os.path.join(base_dir, "dataset", "digit-recogniser", "train-images-idx3-ubyte")
        labels_path = os.path.join(base_dir, "dataset", "digit-recogniser", "train-labels-idx1-ubyte")

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print("MNIST dataset files not found. Downloading and decompressing automatically...")
            from scripts.download_digit_recognizer_dataset import main as download_mnist
            download_mnist()

        # 1. Parse the Binary Image File (IDX3 format)
        # Header structure (16 bytes):
        # - magic number: 4 bytes (integer) -> 2051
        # - number of images: 4 bytes (integer)
        # - number of rows: 4 bytes (integer) -> 28
        # - number of columns: 4 bytes (integer) -> 28
        with open(images_path, "rb") as f:
            # ">IIII" specifies big-endian (>) and 4 unsigned integers (IIII)
            _magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            read_count = min(num_images, limit)
            image_size = rows * cols

            # Read raw pixel bytes for the selected subset of images
            raw_pixels = f.read(read_count * image_size)

        inputs = []
        for i in range(read_count):
            start = i * image_size
            img_bytes = raw_pixels[start : start + image_size]
            # Help Comment:
            # Raw pixel bytes are integers from 0 to 255.
            # Neural networks train much better when inputs are scaled to [0.0, 1.0]
            # to prevent gradients from exploding or saturating the activation functions.
            scaled_img = [float(b) / 255.0 for b in img_bytes]
            inputs.append(scaled_img)

        # 2. Parse the Binary Label File (IDX1 format)
        # Header structure (8 bytes):
        # - magic number: 4 bytes (integer) -> 2049
        # - number of items: 4 bytes (integer)
        with open(labels_path, "rb") as f:
            # ">II" specifies big-endian (>) and 2 unsigned integers (II)
            _magic, num_items = struct.unpack(">II", f.read(8))
            read_count = min(num_items, limit)

            # Read label bytes (each byte is an integer 0-9)
            raw_labels = f.read(read_count)

        targets = []
        for label in raw_labels:
            # Help Comment:
            # Since our network has 10 outputs, we cannot feed raw digit labels (e.g. 5).
            # We must convert each label to a 'one-hot encoded vector' of length 10.
            # Example: label 3 -> [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            one_hot = [0.0] * 10
            one_hot[int(label)] = 1.0
            targets.append(one_hot)

        return inputs, targets

    def get_loss_function(self) -> LossFunction:
        return MeanSquaredError()

    def decode_output(self, raw_prediction: list[float]) -> str:
        # The output of the network is a list of 10 probabilities (one for each digit).
        # We identify the predicted digit by taking the index of the highest probability.
        max_prob = max(raw_prediction)
        predicted_digit = raw_prediction.index(max_prob)
        return f"Digit: {predicted_digit} (Confidence: {max_prob:.2%})"

    def get_default_training_params(self) -> DigitRecogniserTrainingParams:
        # Help Comment:
        # With 1,000 images, running 20 epochs takes a few seconds in pure Python.
        # We set learning rate to 0.1 for steady gradient descent.
        # We set console_frequency to 5 so we get training progress printed every 5 epochs.
        return {"epochs": 20, "learning_rate": 0.1, "console_frequency": 1}
