"""Train and save a NumPy neural network on the MNIST handwritten digits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mnist_data import load_mnist
from neural_network import NeuralNetwork


def train_model(epochs: int, learning_rate: float, hidden_size: int, training_examples: int | None, model_path: Path) -> list[dict[str, float]]:
    """Load MNIST, train a fresh model, save it, and return epoch history."""
    train_features, train_labels, test_features, test_labels = load_mnist()
    if training_examples:
        train_features, train_labels = train_features[:training_examples], train_labels[:training_examples]
    model = NeuralNetwork(hidden_size=hidden_size)
    history = []
    for metric in model.train(train_features, train_labels, epochs, learning_rate, validation_data=(test_features, test_labels)):
        record = {"epoch": metric.epoch, "loss": round(metric.loss, 4), "accuracy": round(metric.accuracy, 4)}
        history.append(record)
        print(f"Epoch {metric.epoch:02d}/{epochs}: test loss={metric.loss:.4f}, test accuracy={metric.accuracy:.2%}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    model_path.with_suffix(".json").write_text(json.dumps({"hidden_size": hidden_size, "learning_rate": learning_rate, "epochs": epochs, "history": history}, indent=2), encoding="utf-8")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--training-examples", type=int, default=None, help="Optional cap useful for quick experiments")
    parser.add_argument("--model-path", type=Path, default=Path("models/mnist_numpy_model.npz"))
    arguments = parser.parse_args()
    train_model(arguments.epochs, arguments.learning_rate, arguments.hidden_size, arguments.training_examples, arguments.model_path)
