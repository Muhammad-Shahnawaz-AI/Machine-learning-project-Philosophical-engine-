import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from app.models import PhilosophicalEngine


class PhilosophicalEngineGUI:
    def __init__(self, engine=None):
        self.engine = engine or PhilosophicalEngine()
        self.root = tk.Tk()
        self.root.title("Philosophical Engine GUI")
        self.root.geometry("1200x700")

        self.dataset_type = tk.StringVar(value="classification")
        self.algorithm = tk.StringVar(value=self.engine.available_modes()[0])
        self.epochs = tk.IntVar(value=120)
        self.learning_rate = tk.DoubleVar(value=0.05)

        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.boundary_ax = self.figure.add_subplot(121)
        self.metric_ax = self.figure.add_subplot(122)

        self.build_interface()
        self.dataset = None
        self.history = {}
        self.generate_dataset()
        self.update_plots()

    def build_interface(self):
        controls = ttk.Frame(self.root, padding="10")
        controls.grid(row=0, column=0, sticky="nsew")

        ttk.Label(controls, text="Mode of Inquiry:", font=("Helvetica", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(controls, self.algorithm, self.engine.available_modes()[0], *self.engine.available_modes()).grid(row=1, column=0, sticky="ew")

        ttk.Label(controls, text="Dataset Type:", font=("Helvetica", 11, "bold")).grid(row=2, column=0, pady=(10, 0), sticky="w")
        ttk.OptionMenu(controls, self.dataset_type, "classification", "classification", "regression").grid(row=3, column=0, sticky="ew")

        ttk.Label(controls, text="Epochs:", font=("Helvetica", 11, "bold")).grid(row=4, column=0, pady=(10, 0), sticky="w")
        ttk.Entry(controls, textvariable=self.epochs).grid(row=5, column=0, sticky="ew")

        ttk.Label(controls, text="Learning Rate:", font=("Helvetica", 11, "bold")).grid(row=6, column=0, pady=(10, 0), sticky="w")
        ttk.Entry(controls, textvariable=self.learning_rate).grid(row=7, column=0, sticky="ew")

        ttk.Button(controls, text="Generate Data", command=self.generate_dataset).grid(row=8, column=0, pady=(15, 5), sticky="ew")
        ttk.Button(controls, text="Train Selected Mode", command=self.train).grid(row=9, column=0, pady=5, sticky="ew")
        ttk.Button(controls, text="Reset View", command=self.reset).grid(row=10, column=0, pady=5, sticky="ew")

        self.verdict_text = scrolledtext.ScrolledText(controls, width=35, height=18, wrap=tk.WORD)
        self.verdict_text.grid(row=11, column=0, pady=(20, 0), sticky="nsew")
        controls.rowconfigure(11, weight=1)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.figure, canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset(self):
        self.dataset_type.set("classification")
        self.algorithm.set(self.engine.available_modes()[0])
        self.epochs.set(120)
        self.learning_rate.set(0.05)
        self.generate_dataset()
        self.update_plots()
        self.verdict_text.configure(state="normal")
        self.verdict_text.delete("1.0", tk.END)
        self.verdict_text.configure(state="disabled")

    def generate_dataset(self):
        kind = self.dataset_type.get()
        self.X, self.y = self.engine.generate_dataset(kind, n_samples=300)
        self.history = {}
        self.update_plots()
        self.display_verdict("New dataset generated. Choose a mode and train to see the philosophical verdict.")

    def train(self):
        mode = self.algorithm.get()
        if mode == "Teleology":
            kind = "regression"
        else:
            kind = "classification"

        if self.dataset_type.get() != kind:
            self.dataset_type.set(kind)
            self.X, self.y = self.engine.generate_dataset(kind, n_samples=300)

        kwargs = {
            "epochs": self.epochs.get(),
            "learning_rate": self.learning_rate.get(),
        }

        try:
            self.history = self.engine.train_model(mode, self.X, self.y, **kwargs)
            self.update_plots(mode=mode)
            verdict = self.engine.verdict(mode, self.history, self.X, self.y)
            self.display_verdict(verdict)
        except Exception as error:
            self.display_verdict(f"Training error: {error}")

    def display_verdict(self, message: str):
        self.verdict_text.configure(state="normal")
        self.verdict_text.delete("1.0", tk.END)
        self.verdict_text.insert(tk.END, message)
        self.verdict_text.configure(state="disabled")

    def update_plots(self, mode=None):
        mode = mode or self.algorithm.get()
        self.boundary_ax.clear()
        self.metric_ax.clear()

        self.boundary_ax.set_title(f"{mode} - Decision Surface")
        self.metric_ax.set_title("Training Curve")

        if self.X is None:
            return

        self.plot_decision_boundary(mode)
        self.plot_dataset(mode)
        self.plot_history()
        self.canvas.draw()

    def plot_dataset(self, mode):
        if self.X.shape[1] != 2:
            return

        if mode == "Taxonomy of Being":
            labels = self.engine.infer(mode, self.X)
            self.boundary_ax.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap="viridis", s=35, edgecolor="k")
            return

        if self.dataset_type.get() == "regression":
            self.boundary_ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap="coolwarm", s=40, edgecolor="k")
        else:
            self.boundary_ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap="Set1", s=40, edgecolor="k")

    def plot_decision_boundary(self, mode):
        if self.X.shape[1] != 2:
            return

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        try:
            predictions = self.engine.infer(mode, grid)
            predictions = np.array(predictions).reshape(xx.shape)
            if self.dataset_type.get() == "regression":
                self.boundary_ax.contourf(xx, yy, predictions, cmap="coolwarm", alpha=0.4)
            else:
                self.boundary_ax.contourf(xx, yy, predictions, cmap="Pastel2", alpha=0.5)
        except Exception:
            pass

    def plot_history(self):
        if not self.history:
            self.metric_ax.text(0.5, 0.5, "Train a model to see the learning curve.", ha="center", va="center", fontsize=12)
            return

        if "loss" in self.history:
            losses = self.history["loss"]
            self.metric_ax.plot(losses, color="#3f6d9d")
            self.metric_ax.set_xlabel("Epoch")
            self.metric_ax.set_ylabel("Loss")
            self.metric_ax.grid(True, linestyle="--", alpha=0.4)
        elif "accuracy" in self.history:
            self.metric_ax.plot(self.history["accuracy"], color="#3f6d9d")
            self.metric_ax.set_xlabel("Epoch")
            self.metric_ax.set_ylabel("Accuracy")
            self.metric_ax.grid(True, linestyle="--", alpha=0.4)
        else:
            self.metric_ax.text(0.5, 0.5, "No training history available.", ha="center", va="center", fontsize=12)

    def run(self):
        self.root.mainloop()
