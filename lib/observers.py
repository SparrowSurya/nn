"""
This module implements the observer pattern to monitor, log, and visualize
training progress and execution steps of the neural network.
"""

from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lib.network import NeuralNetwork


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

    def on_test_model(self, nn: "NeuralNetwork"):
        """Called when manual testing mode is enabled."""
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

    def on_test_model(self, nn: "NeuralNetwork"):
        for observer in self.observers:
            observer.on_test_model(nn)


class XorTestObserver(RunObserver):
    """Observer specifically for manual testing of the XOR model."""

    def on_test_model(self, nn: "NeuralNetwork"):
        print("\n--- Manual Testing (XOR) ---")
        print("Enter input as 00, 01, 10, 11 (or 'exit' to quit):")
        while True:
            try:
                user_input = input("Input: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if len(user_input) != 2 or not all(c in "01" for c in user_input):
                    print("Invalid input format. Use 00, 01, 10, or 11.")
                    continue
                x1 = float(user_input[0])
                x2 = float(user_input[1])
                prediction = nn.forward([x1, x2])
                prob = prediction[0]
                val = round(prob)
                confidence = prob if val == 1 else (1.0 - prob)
                print(f"Output: {val} ({confidence:.0%})")
            except (KeyboardInterrupt, EOFError):
                break


class DigitTestObserver(RunObserver):
    """Observer that launches a Tkinter UI for drawing and testing the digit recogniser."""

    def on_test_model(self, nn: "NeuralNetwork"):
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk

        root = tk.Tk()
        root.title("Digit Recogniser Test")
        root.resizable(False, False)
        
        # Explicit light theme setup (forces light background even if OS is in dark mode)
        BG_COLOR = "#F0F0F0"
        FG_COLOR = "#000000"
        HIGHLIGHT_BORDER = "blue"
        DEFAULT_BORDER = "#F0F0F0"

        root.configure(bg=BG_COLOR)

        # In-memory Pillow image for drawing (280x280, grayscale, initialized to 0/black)
        pil_img = Image.new("L", (280, 280), color=0)
        pil_draw = ImageDraw.Draw(pil_img)

        # Drawing variables
        last_x = [None]
        last_y = [None]
        preview_photo: list[Any] = [None]

        # Draw Title
        title_label = tk.Label(root, text="Draw a digit (0-9)", font=("Arial", 14, "bold"), bg=BG_COLOR, fg=FG_COLOR)
        title_label.pack(pady=10)

        # Frame for side-by-side canvases
        canvas_frame = tk.Frame(root, bg=BG_COLOR)
        canvas_frame.pack(padx=20, pady=5)

        # Left Canvas Column (Drawing)
        left_col = tk.Frame(canvas_frame, bg=BG_COLOR)
        left_col.grid(row=0, column=0, padx=15)
        lbl_left = tk.Label(left_col, text="Input Canvas (Draw here)", font=("Arial", 11, "bold"), bg=BG_COLOR, fg=FG_COLOR)
        lbl_left.pack(pady=2)
        canvas = tk.Canvas(left_col, width=280, height=280, bg="black", highlightthickness=1, highlightbackground="gray")
        canvas.pack()

        # Right Canvas Column (Downscaled Preview)
        right_col = tk.Frame(canvas_frame, bg=BG_COLOR)
        right_col.grid(row=0, column=1, padx=15)
        lbl_right = tk.Label(right_col, text="Network Input Preview (28x28)", font=("Arial", 11, "bold"), bg=BG_COLOR, fg=FG_COLOR)
        lbl_right.pack(pady=2)
        preview_canvas = tk.Canvas(right_col, width=280, height=280, bg="black", highlightthickness=1, highlightbackground="gray")
        preview_canvas.pack()

        # Drawing callbacks
        def on_press(event):
            last_x[0] = event.x
            last_y[0] = event.y

        def on_drag(event):
            if last_x[0] is not None and last_y[0] is not None:
                # Draw on Tkinter canvas
                canvas.create_line(
                    last_x[0], last_y[0], event.x, event.y,
                    fill="white", width=14, capstyle=tk.ROUND, smooth=True
                )
                # Draw on Pillow Image Draw
                pil_draw.line(
                    [last_x[0], last_y[0], event.x, event.y],
                    fill=255, width=14
                )
            last_x[0] = event.x
            last_y[0] = event.y

        def on_release(event):
            last_x[0] = None
            last_y[0] = None

        canvas.bind("<Button-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        # Result display frames
        result_frame = tk.Frame(root, bg=BG_COLOR)
        result_frame.pack(pady=15)

        # Columns for 0-9
        col_frames = []
        digit_labels = []
        prob_labels = []

        for digit in range(10):
            # A frame for each digit column to outline the selection cleanly
            col_frame = tk.Frame(
                result_frame,
                borderwidth=1,
                relief=tk.SOLID,
                highlightthickness=3,
                highlightbackground=DEFAULT_BORDER,
                bg=BG_COLOR
            )
            col_frame.grid(row=0, column=digit, padx=4)
            col_frames.append(col_frame)
            
            lbl_digit = tk.Label(col_frame, text=str(digit), font=("Arial", 16, "bold"), bg=BG_COLOR, fg=FG_COLOR)
            lbl_digit.pack(pady=2)
            digit_labels.append(lbl_digit)
            
            lbl_prob = tk.Label(col_frame, text="0.0%", font=("Arial", 12), bg=BG_COLOR, fg=FG_COLOR)
            lbl_prob.pack(pady=2)
            prob_labels.append(lbl_prob)

        # Button Frame
        btn_frame = tk.Frame(root, bg=BG_COLOR)
        btn_frame.pack(pady=15)

        def clear_canvas():
            canvas.delete("all")
            preview_canvas.delete("all")
            preview_photo[0] = None
            # Reset PIL image
            pil_draw.rectangle([0, 0, 280, 280], fill=0)
            # Reset labels & outlines
            for d in range(10):
                col_frames[d].config(highlightbackground=DEFAULT_BORDER)
                prob_labels[d].config(text="0.0%")

        def predict_digit():
            predict_btn.config(state=tk.DISABLED)
            clear_btn.config(state=tk.DISABLED)
            root.update_idletasks()

            try:
                # 1. Resize drawn image to 28x28 using Lanczos
                resized = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # 2. Display the 28x28 preview upscaled back to 280x280 using Nearest Neighbor
                preview_img = resized.resize((280, 280), Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(preview_img)
                preview_photo[0] = photo
                preview_canvas.delete("all")
                preview_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

                # 3. Scale pixel values to [0.0, 1.0]
                pixels = list(resized.getdata())  # type: ignore
                scaled = [float(p) / 255.0 for p in pixels]

                # 4. Run prediction through NeuralNetwork
                outputs = nn.forward(scaled)

                max_prob = max(outputs)
                max_idx = outputs.index(max_prob)

                # Update UI elements
                for d in range(10):
                    prob = outputs[d]
                    prob_labels[d].config(text=f"{prob:.1%}")
                    if d == max_idx:
                        col_frames[d].config(highlightbackground=HIGHLIGHT_BORDER)
                    else:
                        col_frames[d].config(highlightbackground=DEFAULT_BORDER)
            finally:
                predict_btn.config(state=tk.NORMAL)
                clear_btn.config(state=tk.NORMAL)

        clear_btn = tk.Button(btn_frame, text="Clear", command=clear_canvas, font=("Arial", 12), width=10)
        clear_btn.grid(row=0, column=0, padx=15)

        predict_btn = tk.Button(btn_frame, text="Predict", command=predict_digit, font=("Arial", 12), width=10)
        predict_btn.grid(row=0, column=1, padx=15)

        root.mainloop()
