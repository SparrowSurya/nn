"""
This file is the main entry point of the program.
"""

import random
from lib.activations import ReLU, Sigmoid
from lib.losses import MeanSquaredError
from lib.observers import ConsoleEpochObserver, PlotEpochObserver, CompositeEpochObserver
from lib.layers import NeuralLayer
from lib.network import NeuralNetwork


def main():
    # Helper for weight initialization between -1.0 and 1.0
    random_weight_init = lambda: random.uniform(-1.0, 1.0)
    random.seed(42)

    nn = NeuralNetwork()
    nn.add_layer(NeuralLayer.from_size(2, 3, ReLU(), random_func=random_weight_init))
    nn.add_layer(NeuralLayer.from_size(3, 1, Sigmoid(), random_func=random_weight_init))

    if not nn.validate():
        print("Invalid neural network configuration.")
        return

    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]

    print("--- Predictions BEFORE training ---")
    for x in inputs:
        prediction = nn.forward(x)
        print(f"Input: {x} -> Prediction: {prediction}")

    print("\n--- Training ---")
    # Setup modular observers
    console_observer = ConsoleEpochObserver(frequency=2000)
    plot_observer = PlotEpochObserver()
    composite_observer = CompositeEpochObserver([console_observer, plot_observer])

    nn.train(
        inputs,
        targets,
        epochs=10000,
        learning_rate=0.2,
        loss_fn=MeanSquaredError(),
        observer=composite_observer,
    )

    print("\n--- Predictions AFTER training ---")
    for x in inputs:
        prediction = nn.forward(x)
        print(f"Input: {x} -> Prediction: {prediction}")

    # Plot the results
    plot_observer.plot()


if __name__ == "__main__":
    main()
