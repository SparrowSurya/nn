"""
This module implements the interactive Tkinter-based GUI wizard
for training, visualizing, and testing neural networks.
It provides a customizable base class that task programs can extend.
"""

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from lib.program import NeuralNetworkProgram
from lib.network import NeuralNetwork
from lib.visualizations import visualize_network_structure
from lib.observers import RunObserver
from typing import Any, Callable


class TrainingStopException(Exception):
    """Custom exception raised when training is interrupted by the user."""
    pass


class NeuralNetworkGui:
    """Base Tkinter-based wizard UI for step-by-step neural network training and testing."""

    def __init__(self, task: NeuralNetworkProgram[Any, Any, Any]):
        self.task = task
        self.nn: NeuralNetwork = task.build_network()
        self.inputs: list[list[float]] = []
        self.targets: list[list[float]] = []
        self.losses: list[float] = []

        # Load dataset
        self.inputs, self.targets = self.task.load_training_data()

        # GUI Constants
        self.BG_COLOR = "#F0F0F0"
        self.FG_COLOR = "#000000"
        self.HIGHLIGHT_BORDER = "blue"
        self.DEFAULT_BORDER = "#F0F0F0"

        # Initialize Main Window
        self.root = tk.Tk()
        self.root.title(f"Neural Network Wizard - {self.task.name}")
        self.root.geometry("950x700")
        self.root.configure(bg=self.BG_COLOR)
        self.root.resizable(True, True)

        # Allow window content to stretch
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.columnconfigure(0, weight=1)

        # Current state
        self.current_phase = 1
        self.trained = False
        self.training_stopped = False

        # Main Layout Frames
        self.main_container = tk.Frame(self.root, bg=self.BG_COLOR)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)

        # Configure columns for left/right split
        self.main_container.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.columnconfigure(1, weight=1)

        self.nav_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        self.nav_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=15)

        self.setup_navigation()
        self.show_phase(1)

    def setup_navigation(self):
        """Sets up the bottom navigation wizard bar."""
        self.prev_btn = tk.Button(self.nav_frame, text="< Previous Phase", font=("Arial", 12), command=self.go_prev)
        self.prev_btn.pack(side=tk.LEFT)

        self.phase_label = tk.Label(self.nav_frame, text="", font=("Arial", 12, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR)
        self.phase_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.next_btn = tk.Button(self.nav_frame, text="Next Phase >", font=("Arial", 12), command=self.go_next)
        self.next_btn.pack(side=tk.RIGHT)

    def update_navigation_buttons(self):
        """Updates navigation text and enablement state depending on phase."""
        self.prev_btn.config(state=tk.NORMAL if self.current_phase > 1 else tk.DISABLED)
        
        # Block going to Phase 3 if the model is not trained yet
        if self.current_phase == 2 and not self.trained:
            self.next_btn.config(state=tk.DISABLED)
        else:
            self.next_btn.config(state=tk.NORMAL if self.current_phase < 3 else tk.DISABLED)

        phase_titles = {
            1: "Phase 1: Dataset Browsing & Untrained Model Inspection",
            2: "Phase 2: Hyperparameter Configuration & Training",
            3: "Phase 3: Interactive Manual Testing & Model Utilities"
        }
        self.phase_label.config(text=phase_titles[self.current_phase])

    def go_prev(self):
        if self.current_phase > 1:
            self.show_phase(self.current_phase - 1)

    def go_next(self):
        if self.current_phase < 3:
            if self.current_phase == 2 and not self.trained:
                messagebox.showwarning("Training Required", "Please train the model before proceeding to testing.")
                return
            self.show_phase(self.current_phase + 1)

    def show_phase(self, phase_num: int):
        """Clears the main container and builds widgets for the selected phase."""
        self.current_phase = phase_num
        
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()

        self.update_navigation_buttons()

        if phase_num == 1:
            self.build_phase_1()
        elif phase_num == 2:
            self.build_phase_2()
        elif phase_num == 3:
            self.build_phase_3()

    # =========================================================================
    # TASK OVERRIDABLE CUSTOMIZATIONS (DEFAULT IMPLEMENTATIONS)
    # =========================================================================
    def build_dataset_input_widget(self, parent: Any) -> tuple[Any, Callable[[list[float]], None]]:
        """Builds and returns the dataset input browser widget and its update callback."""
        lbl = tk.Label(parent, text="", font=("Arial", 12), bg=self.BG_COLOR, fg=self.FG_COLOR, wraplength=400)
        lbl.pack(pady=20)
        
        def update_fn(x: list[float]):
            lbl.config(text=f"Input Vector:\n{x}")
            
        return lbl, update_fn

    def build_manual_input_widget(self, parent: Any, on_predict: Callable[[list[float]], None]) -> Any:
        """Builds and returns the manual test input widget."""
        frame = tk.Frame(parent, bg=self.BG_COLOR)
        frame.pack(pady=10)
        
        tk.Label(frame, text="Input vector (comma or space separated):", font=("Arial", 11), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(pady=5)
        entry = tk.Entry(frame, font=("Arial", 11), width=25)
        entry.pack(pady=5)
        
        def trigger_predict():
            try:
                parts = entry.get().replace(",", " ").split()
                x = [float(v) for v in parts]
                if len(x) != len(self.inputs[0]):
                    raise ValueError
                on_predict(x)
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter exactly {len(self.inputs[0])} numbers.")
                
        btn = tk.Button(frame, text="Predict Custom Input", font=("Arial", 11), command=trigger_predict)
        btn.pack(pady=10)
        return frame

    def build_prediction_output_widget(self, parent: Any) -> tuple[Any, Callable[[list[float]], None]]:
        """Builds and returns the output probability bars and their update callback."""
        frame = tk.Frame(parent, bg=self.BG_COLOR)
        frame.pack(pady=10, fill=tk.X, expand=True)
        
        num_outputs = len(self.targets[0])
        # If single output, it represents a binary probability (Class 0 vs Class 1)
        num_classes = 2 if num_outputs == 1 else num_outputs
        
        rows = []
        bar_canvases = []
        bar_rects = []
        prob_labels = []
        
        # Colors for graphical bars
        COLOR_HIGHLIGHT = "blue"
        COLOR_NORMAL = "gray"
        
        for idx in range(num_classes):
            row_frame = tk.Frame(frame, bg=self.BG_COLOR)
            row_frame.pack(fill=tk.X, pady=3, expand=True)
            rows.append(row_frame)
            
            # Display name: digit or Class index
            display_name = str(idx) if num_classes > 2 else f"Class {idx}"
            lbl_digit = tk.Label(row_frame, text=display_name, font=("Arial", 11, "bold"), width=8, anchor="w", bg=self.BG_COLOR, fg=self.FG_COLOR)
            lbl_digit.pack(side=tk.LEFT, padx=5)
            
            # Bar canvas
            canvas = tk.Canvas(row_frame, width=200, height=16, bg="white", highlightthickness=1, highlightbackground="lightgray")
            canvas.pack(side=tk.LEFT, padx=5)
            bar_canvases.append(canvas)
            
            # Draw a temporary empty bar
            rect = canvas.create_rectangle(0, 0, 0, 16, fill=COLOR_NORMAL, width=0)
            bar_rects.append(rect)
            
            # Probability percentage
            lbl_prob = tk.Label(row_frame, text="0.0%", font=("Arial", 11), width=6, anchor="e", bg=self.BG_COLOR, fg=self.FG_COLOR)
            lbl_prob.pack(side=tk.LEFT, padx=5)
            prob_labels.append(lbl_prob)
            
        def update_fn(outputs: list[float]):
            if len(outputs) == 1:
                prob = outputs[0]
                probs = [1.0 - prob, prob]
            else:
                probs = outputs
                
            max_prob = max(probs)
            max_idx = probs.index(max_prob)
            
            for d in range(len(probs)):
                p_val = probs[d]
                prob_labels[d].config(text=f"{p_val:.1%}")
                
                canvas = bar_canvases[d]
                rect = bar_rects[d]
                bar_width = int(p_val * 200)
                
                canvas.coords(rect, 0, 0, bar_width, 16)
                
                if d == max_idx:
                    canvas.itemconfig(rect, fill=COLOR_HIGHLIGHT)
                    prob_labels[d].config(fg=COLOR_HIGHLIGHT, font=("Arial", 11, "bold"))
                else:
                    canvas.itemconfig(rect, fill=COLOR_NORMAL)
                    prob_labels[d].config(fg=self.FG_COLOR, font=("Arial", 11))
                    
        return frame, update_fn

    # =========================================================================
    # PHASE 1: PRE-TRAINING INSPECTION
    # =========================================================================
    def build_phase_1(self):
        # Two-column grid layout for resizing
        left_panel = tk.Frame(self.main_container, bg=self.BG_COLOR)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10)
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)

        right_panel = tk.Frame(self.main_container, bg=self.BG_COLOR)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=10)
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        # Left: Data Browser
        lbl_browser = tk.Label(left_panel, text="Training Data Browser", font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl_browser.pack(pady=10)

        # Construct specific dataset input widget from subclass/base
        browser_content_frame = tk.Frame(left_panel, bg=self.BG_COLOR)
        browser_content_frame.pack(fill=tk.BOTH, expand=True)
        _, update_input_widget = self.build_dataset_input_widget(browser_content_frame)

        lbl_target = tk.Label(left_panel, text="", font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl_target.pack(pady=2)

        lbl_predicted = tk.Label(left_panel, text="", font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg="blue")
        lbl_predicted.pack(pady=2)

        # Right: Untrained Predictions
        lbl_pred = tk.Label(right_panel, text="Untrained Model Probabilities", font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl_pred.pack(pady=10)

        # Construct specific prediction output columns
        pred_content_frame = tk.Frame(right_panel, bg=self.BG_COLOR)
        pred_content_frame.pack(fill=tk.BOTH, expand=True)
        _, update_output_widget = self.build_prediction_output_widget(pred_content_frame)

        def update_inspection(idx_str: str):
            try:
                idx = int(idx_str)
            except ValueError:
                return

            if idx < 0 or idx >= len(self.inputs):
                return

            x = self.inputs[idx]
            y = self.targets[idx]

            expected_decoded = self.task.decode_target(y)
            lbl_target.config(text=f"Expected Target: {expected_decoded}")
            
            # Delegate rendering & predictions to task callbacks
            update_input_widget(x)
            outputs = self.nn.forward(x)
            update_output_widget(outputs)
            
            pred_decoded = self.task.decode_target(outputs)
            lbl_predicted.config(text=f"Model Prediction: {pred_decoded}")

        # Slider frame
        slider_frame = tk.Frame(left_panel, bg=self.BG_COLOR)
        slider_frame.pack(pady=15)

        tk.Label(slider_frame, text="Dataset Index:", font=("Arial", 12), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT, padx=5)
        
        spin = tk.Spinbox(
            slider_frame,
            from_=0,
            to=len(self.inputs) - 1,
            width=8,
            font=("Arial", 12),
            command=lambda: update_inspection(spin.get())
        )
        spin.pack(side=tk.LEFT, padx=5)

        tk.Label(slider_frame, text=f"/ {len(self.inputs)}", font=("Arial", 12), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT, padx=5)

        update_inspection("0")

    # =========================================================================
    # PHASE 2: TRAINING MODEL
    # =========================================================================
    def build_phase_2(self):
        left_panel = tk.Frame(self.main_container, bg=self.BG_COLOR)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10)

        right_panel = tk.Frame(self.main_container, bg=self.BG_COLOR)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=10)
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        # Left: Inputs
        lbl_cfg = tk.Label(left_panel, text="Training Configuration", font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl_cfg.pack(pady=10)

        params = self.task.get_default_training_params()

        tk.Label(left_panel, text="Epochs:", font=("Arial", 12), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(pady=5)
        epoch_entry = tk.Entry(left_panel, font=("Arial", 12), width=15)
        epoch_entry.insert(0, str(params.get("epochs", 10)))
        epoch_entry.pack(pady=2)

        tk.Label(left_panel, text="Learning Rate:", font=("Arial", 12), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(pady=5)
        lr_entry = tk.Entry(left_panel, font=("Arial", 12), width=15)
        lr_entry.insert(0, str(params.get("learning_rate", 0.1)))
        lr_entry.pack(pady=2)

        progress_lbl = tk.Label(left_panel, text="Model Status: Untrained", font=("Arial", 12, "italic"), bg=self.BG_COLOR, fg="darkorange")
        if self.trained:
            progress_lbl.config(text="Model Status: Trained!", fg="green")
        progress_lbl.pack(pady=20)

        # Right: Embedded Matplotlib Plot
        lbl_plot = tk.Label(right_panel, text="Loss Progression Chart", font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl_plot.pack(pady=10)

        fig = Figure(figsize=(4.5, 3.2), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title("Loss vs. Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)

        canvas_plot = FigureCanvasTkAgg(fig, master=right_panel)
        canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if self.trained and self.losses:
            ax.plot(self.losses, color="blue", linewidth=2)
            canvas_plot.draw()

        def start_training():
            try:
                epochs = int(epoch_entry.get())
                lr = float(lr_entry.get())
                if epochs <= 0 or lr <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Parameters", "Please enter positive integer for epochs and positive float for learning rate.")
                return

            epoch_entry.config(state=tk.DISABLED)
            lr_entry.config(state=tk.DISABLED)
            train_btn.config(state=tk.DISABLED)
            stop_btn.config(state=tk.NORMAL)
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.root.update_idletasks()

            self.nn = self.task.build_network()
            self.losses = []
            self.training_stopped = False

            outer_self = self

            class GuiTrainingObserver(RunObserver):
                def on_epoch_end(self, epoch: int, total_epochs: int, loss: float):
                    if outer_self.training_stopped:
                        raise TrainingStopException("Training stopped by user.")
                    outer_self.losses.append(loss)
                    progress_lbl.config(text=f"Training: Epoch {epoch + 1}/{total_epochs} (Loss: {loss:.4f})", fg="blue")
                    ax.clear()
                    ax.plot(outer_self.losses, color="blue", linewidth=2)
                    ax.set_title("Training Loss Progression")
                    ax.set_xlabel("Epochs")
                    ax.set_ylabel("Loss")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    canvas_plot.draw()
                    outer_self.root.update()

            try:
                self.nn.train(
                    self.inputs,
                    self.targets,
                    epochs=epochs,
                    learning_rate=lr,
                    loss_fn=self.task.get_loss_function(),
                    observer=GuiTrainingObserver()
                )
                self.trained = True
                progress_lbl.config(text="Model Status: Trained!", fg="green")
            except TrainingStopException:
                self.trained = True
                progress_lbl.config(text="Model Status: Training Stopped (Partial model ready!)", fg="red")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during training:\n{e}")
            finally:
                epoch_entry.config(state=tk.NORMAL)
                lr_entry.config(state=tk.NORMAL)
                train_btn.config(state=tk.NORMAL)
                stop_btn.config(state=tk.DISABLED)
                self.update_navigation_buttons()

        btn_frame = tk.Frame(left_panel, bg=self.BG_COLOR)
        btn_frame.pack(pady=10)

        train_btn = tk.Button(btn_frame, text="Train Model", font=("Arial", 12, "bold"), command=start_training, width=15)
        train_btn.grid(row=0, column=0, padx=5)

        def stop_training():
            self.training_stopped = True
            stop_btn.config(state=tk.DISABLED)
            progress_lbl.config(text="Stopping training...", fg="red")

        stop_btn = tk.Button(btn_frame, text="Stop Training", font=("Arial", 12, "bold"), command=stop_training, width=15, state=tk.DISABLED)
        stop_btn.grid(row=0, column=1, padx=5)

    # =========================================================================
    # PHASE 3: INTERACTIVE TESTING & UTILITIES
    # =========================================================================
    def build_phase_3(self):
        left_panel = tk.Frame(self.main_container, bg=self.BG_COLOR)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10)

        right_panel = tk.Frame(self.main_container, bg=self.BG_COLOR)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=10)

        # Output probability bars (shared between manual testing and dataset browsing)
        result_frame = tk.Frame(right_panel, bg=self.BG_COLOR)
        result_frame.pack(pady=15)
        _, update_output_widget = self.build_prediction_output_widget(result_frame)

        # ---------------------------------------------------------------------
        # Left Panel: Manual Testing Mode
        # ---------------------------------------------------------------------
        lbl_manual = tk.Label(left_panel, text="Manual Test Input", font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl_manual.pack(pady=5)

        lbl_manual_predicted = tk.Label(left_panel, text="", font=("Arial", 12, "bold"), bg=self.BG_COLOR, fg="blue")
        lbl_manual_predicted.pack(pady=5)

        def on_manual_predict(x: list[float]):
            outputs = self.nn.forward(x)
            update_output_widget(outputs)
            pred_decoded = self.task.decode_target(outputs)
            lbl_manual_predicted.config(text=f"Model Prediction: {pred_decoded}")

        # Delegate building the manual input widget to the subclass/base
        self.build_manual_input_widget(left_panel, on_manual_predict)

        # ---------------------------------------------------------------------
        # Right Panel: Dataset Testing & Extra Utilities
        # ---------------------------------------------------------------------
        tk.Label(right_panel, text="Test Model on Dataset Sample", font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(pady=5)

        dataset_test_frame = tk.Frame(right_panel, bg=self.BG_COLOR)
        dataset_test_frame.pack(pady=5)

        tk.Label(dataset_test_frame, text="Browse Index:", font=("Arial", 11), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT, padx=5)

        test_display_frame = tk.Frame(right_panel, bg=self.BG_COLOR)
        test_display_frame.pack(pady=5)
        _, update_test_input_widget = self.build_dataset_input_widget(test_display_frame)

        lbl_test_expected = tk.Label(right_panel, text="", font=("Arial", 12, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl_test_expected.pack(pady=2)

        lbl_test_predicted = tk.Label(right_panel, text="", font=("Arial", 12, "bold"), bg=self.BG_COLOR, fg="blue")
        lbl_test_predicted.pack(pady=2)

        def update_test_prediction(idx_str: str):
            try:
                idx = int(idx_str)
            except ValueError:
                return

            if idx < 0 or idx >= len(self.inputs):
                return

            x = self.inputs[idx]
            y = self.targets[idx]

            expected_decoded = self.task.decode_target(y)
            lbl_test_expected.config(text=f"Expected Target: {expected_decoded}")
            
            # Update inputs display and calculate predictions
            update_test_input_widget(x)
            outputs = self.nn.forward(x)
            update_output_widget(outputs)

            pred_decoded = self.task.decode_target(outputs)
            lbl_test_predicted.config(
                text=f"Model Prediction: {pred_decoded}",
                fg="green" if pred_decoded == expected_decoded else "red"
            )

        spin_test = tk.Spinbox(
            dataset_test_frame,
            from_=0,
            to=len(self.inputs) - 1,
            width=8,
            font=("Arial", 11),
            command=lambda: update_test_prediction(spin_test.get())
        )
        spin_test.pack(side=tk.LEFT, padx=5)

        tk.Label(dataset_test_frame, text=f"/ {len(self.inputs)}", font=("Arial", 11), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side=tk.LEFT, padx=5)

        # ------------------- Extra Utilities -------------------
        utils_frame = tk.Frame(right_panel, bg=self.BG_COLOR)
        utils_frame.pack(pady=20)

        def show_loss_window():
            if not self.losses:
                messagebox.showinfo("No Data", "No training loss history available.")
                return

            plt.figure("Training Loss History")
            plt.plot(self.losses, color="blue", linewidth=2)
            plt.title("Loss Progression Over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--")
            plt.show()

        def show_network_structure():
            visualize_network_structure(self.nn)

        tk.Button(utils_frame, text="Show Loss Graph", font=("Arial", 11), command=show_loss_window, width=20).pack(pady=5)
        tk.Button(utils_frame, text="Show Network Structure", font=("Arial", 11), command=show_network_structure, width=20).pack(pady=5)
        tk.Label(utils_frame, text="This may be slow for large neural networks", font=("Arial", 9, "italic"), bg=self.BG_COLOR, fg="darkred").pack(pady=2)

        update_test_prediction("0")

    def run(self):
        """Starts the Tkinter main loop."""
        self.root.mainloop()
