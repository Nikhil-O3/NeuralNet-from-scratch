"""A small, fully-connected neural network written only with NumPy.

The implementation intentionally exposes every major training step: parameter
initialisation, forward propagation, cross-entropy loss, backpropagation, and
mini-batch gradient descent.  No deep-learning framework is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


@dataclass
class TrainingMetrics:
    """Metrics recorded after one training epoch."""

    epoch: int
    loss: float
    accuracy: float


class NeuralNetwork:
    """A one-hidden-layer classifier using ReLU and softmax activations."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10, seed: int = 42):
        """Create trainable weights with He initialisation for ReLU layers."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        generator = np.random.default_rng(seed)
        self.weights_input_hidden = (
            generator.standard_normal((input_size, hidden_size)).astype(np.float32)
            * np.sqrt(2.0 / input_size)
        )
        self.bias_hidden = np.zeros((1, hidden_size), dtype=np.float32)
        self.weights_hidden_output = (
            generator.standard_normal((hidden_size, output_size)).astype(np.float32)
            * np.sqrt(2.0 / hidden_size)
        )
        self.bias_output = np.zeros((1, output_size), dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Return numerically stable softmax probabilities for each row."""
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run forward propagation and return probabilities plus cached values."""
        hidden_linear = features @ self.weights_input_hidden + self.bias_hidden
        hidden_activation = np.maximum(0, hidden_linear)  # ReLU
        output_linear = hidden_activation @ self.weights_hidden_output + self.bias_output
        probabilities = self._softmax(output_linear)
        return probabilities, (hidden_linear, hidden_activation, probabilities)

    @staticmethod
    def _one_hot(labels: np.ndarray, number_of_classes: int) -> np.ndarray:
        """Convert integer class labels into one-hot encoded target rows."""
        targets = np.zeros((labels.size, number_of_classes), dtype=np.float32)
        targets[np.arange(labels.size), labels] = 1.0
        return targets

    def loss(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Calculate mean categorical cross-entropy loss."""
        clipped = np.clip(probabilities[np.arange(labels.size), labels], 1e-9, 1.0)
        return float(-np.mean(np.log(clipped)))

    def backward(self, features: np.ndarray, labels: np.ndarray, cache: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
        """Use the chain rule to compute gradients for every parameter."""
        hidden_linear, hidden_activation, probabilities = cache
        batch_size = features.shape[0]
        output_gradient = (probabilities - self._one_hot(labels, self.output_size)) / batch_size

        gradient_weights_hidden_output = hidden_activation.T @ output_gradient
        gradient_bias_output = np.sum(output_gradient, axis=0, keepdims=True)
        hidden_gradient = (output_gradient @ self.weights_hidden_output.T) * (hidden_linear > 0)
        gradient_weights_input_hidden = features.T @ hidden_gradient
        gradient_bias_hidden = np.sum(hidden_gradient, axis=0, keepdims=True)
        return {
            "weights_input_hidden": gradient_weights_input_hidden,
            "bias_hidden": gradient_bias_hidden,
            "weights_hidden_output": gradient_weights_hidden_output,
            "bias_output": gradient_bias_output,
        }

    def apply_gradients(self, gradients: dict[str, np.ndarray], learning_rate: float) -> None:
        """Update each parameter with one gradient-descent step."""
        self.weights_input_hidden -= learning_rate * gradients["weights_input_hidden"]
        self.bias_hidden -= learning_rate * gradients["bias_hidden"]
        self.weights_hidden_output -= learning_rate * gradients["weights_hidden_output"]
        self.bias_output -= learning_rate * gradients["bias_output"]

    def train(self, features: np.ndarray, labels: np.ndarray, epochs: int, learning_rate: float, batch_size: int = 128, validation_data: tuple[np.ndarray, np.ndarray] | None = None, on_batch_end: Callable[[int, int], None] | None = None) -> Iterable[TrainingMetrics]:
        """Train with mini-batches, optionally reporting each changed-weight update."""
        generator = np.random.default_rng(7)
        for epoch in range(1, epochs + 1):
            batches = np.array_split(generator.permutation(features.shape[0]), max(1, int(np.ceil(features.shape[0] / batch_size))))
            for batch_number, indices in enumerate(batches, start=1):
                batch_features, batch_labels = features[indices], labels[indices]
                _, cache = self.forward(batch_features)
                self.apply_gradients(self.backward(batch_features, batch_labels, cache), learning_rate)
                if on_batch_end:
                    on_batch_end(epoch, batch_number)
            evaluated_features, evaluated_labels = validation_data or (features, labels)
            probabilities, _ = self.forward(evaluated_features)
            yield TrainingMetrics(epoch, self.loss(probabilities, evaluated_labels), float(np.mean(np.argmax(probabilities, axis=1) == evaluated_labels)))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return one probability distribution for every feature row."""
        probabilities, _ = self.forward(features)
        return probabilities

    def save(self, path: str | Path) -> None:
        """Persist architecture and learned NumPy arrays in a compressed file."""
        np.savez_compressed(path, input_size=self.input_size, hidden_size=self.hidden_size, output_size=self.output_size, weights_input_hidden=self.weights_input_hidden, bias_hidden=self.bias_hidden, weights_hidden_output=self.weights_hidden_output, bias_output=self.bias_output)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        """Reconstruct a trained network from a file created by :meth:`save`."""
        with np.load(path) as data:
            model = cls(int(data["input_size"]), int(data["hidden_size"]), int(data["output_size"]))
            model.weights_input_hidden = data["weights_input_hidden"]
            model.bias_hidden = data["bias_hidden"]
            model.weights_hidden_output = data["weights_hidden_output"]
            model.bias_output = data["bias_output"]
        return model
