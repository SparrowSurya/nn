import argparse
import random
import sys
import importlib
import inspect
from typing import Any
from lib.program import NeuralNetworkProgram


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

    # Instantiate and run the task program with optional visualization settings
    task = program_class()
    task.run(show_loss=args.show_loss, show_network=args.show_network)


if __name__ == "__main__":
    main()
