import argparse
import random
import sys
import importlib
import inspect
from typing import Any
from lib.program import NeuralNetworkProgram
from lib.observers import ConsoleRunObserver, PlotRunObserver, CompositeRunObserver, RunObserver, XorTestObserver, DigitTestObserver


def get_program_class(program_name: str) -> type[NeuralNetworkProgram[Any, Any, Any]] | None:
    """Dynamically imports the program module and searches for a NeuralNetworkProgram subclass."""
    module_name = program_name.lower()
    try:
        # Dynamically import programs.<module_name>
        module_path = f"programs.{module_name}"
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Error: Program module '{module_name}' not found in the 'programs' folder.")
        print(f"Details: {e}")
        print(f"Please check that 'programs/{module_name}.py' exists.")
        return None

    # Inspect module classes to find any class inheriting from NeuralNetworkProgram
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, NeuralNetworkProgram) and obj is not NeuralNetworkProgram:
            return obj

    print(f"Error: No subclass of NeuralNetworkProgram found in 'programs.{module_name}'.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Neural Network Task CLI Runner")
    parser.add_argument(
        "program",
        type=str,
        help="Name of the program to run (e.g. 'xor' to run programs/xor.py)"
    )
    parser.add_argument(
        "--show-loss",
        action="store_true",
        help="Display the training loss progression graph at the end of training"
    )
    parser.add_argument(
        "--show-network",
        action="store_true",
        help="Display the neural network weights and biases visualization diagram"
    )
    parser.add_argument(
        "--manual-testing",
        action="store_true",
        help="Enable interactive manual testing of the trained model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    program_class = get_program_class(args.program)
    if program_class is None:
        sys.exit(1)

    # Instantiate the task program
    task = program_class()

    # Build the observers in main.py
    params = task.get_default_training_params()
    frequency = int(params.get("console_frequency", 2000))
    console_observer = ConsoleRunObserver(frequency=frequency)
    plot_observer = PlotRunObserver() if args.show_loss else None

    # Package into a single CompositeRunObserver
    observers: list[RunObserver] = [console_observer]
    if plot_observer is not None:
        observers.append(plot_observer)
    if args.manual_testing:
        if args.program.lower() == "xor":
            observers.append(XorTestObserver())
        elif args.program.lower() == "digit_recogniser":
            observers.append(DigitTestObserver())

    composite_observer = CompositeRunObserver(observers)

    # Run the task program with the composite observer
    task.run(
        composite_observer,
        show_network=args.show_network,
        manual_testing=args.manual_testing
    )

    # Plot the loss curve if the observer was constructed
    if plot_observer is not None:
        plot_observer.plot()


if __name__ == "__main__":
    main()
