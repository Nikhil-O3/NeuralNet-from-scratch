"""Flask server for drawing MNIST digits and experimenting with NumPy training."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from mnist_data import load_mnist
from neural_network import NeuralNetwork

MODEL_PATH = Path("models/mnist_numpy_model.npz")
app = Flask(__name__)
model_lock = threading.Lock()
model: NeuralNetwork | None = None
training_state: dict[str, object] = {"running": False, "history": [], "message": "No experiment is running."}


def get_model() -> NeuralNetwork:
    """Load the locally trained model once, raising a clear setup error if absent."""
    global model
    with model_lock:
        if model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError("No trained model found. Run `python train.py` first.")
            model = NeuralNetwork.load(MODEL_PATH)
        return model


def _group_input_values(values: np.ndarray) -> list[float]:
    """Reduce 784 MNIST pixels to sixteen 7x7 cells for a readable diagram."""
    return values.reshape(28, 28).reshape(4, 7, 4, 7).mean(axis=(1, 3)).reshape(-1).round(4).tolist()


def model_visualization(network: NeuralNetwork, pixels: np.ndarray | None = None) -> dict[str, object]:
    """Create a compact, truthful view of the current learned weight matrices.

    A 16-cell input grid is used instead of 784 nodes.  Small hidden layers are
    shown exactly; large layers use the product W1 @ W2 as a compact influence
    map, so the browser never claims to display neurons it has omitted.
    """
    visible_hidden = network.hidden_size if network.hidden_size <= 24 else 0
    input_activity = _group_input_values(pixels) if pixels is not None else [0.0] * 16
    if visible_hidden:
        input_weights = network.weights_input_hidden.reshape(4, 7, 4, 7, network.hidden_size).mean(axis=(1, 3))
        payload: dict[str, object] = {
            "mode": "expanded",
            "hidden_size": network.hidden_size,
            "input_to_hidden": input_weights.reshape(16, network.hidden_size).round(4).tolist(),
            "hidden_to_output": network.weights_hidden_output.round(4).tolist(),
            "hidden_activity": [0.0] * network.hidden_size,
        }
        if pixels is not None:
            probabilities, (_, hidden_values, _) = network.forward(pixels.reshape(1, 784))
            payload["hidden_activity"] = hidden_values[0].round(4).tolist()
            payload["output_activity"] = probabilities[0].round(4).tolist()
    else:
        effective_weights = network.weights_input_hidden @ network.weights_hidden_output
        input_to_output = effective_weights.reshape(4, 7, 4, 7, 10).mean(axis=(1, 3))
        payload = {
            "mode": "compact",
            "hidden_size": network.hidden_size,
            "input_to_output": input_to_output.reshape(16, 10).round(4).tolist(),
        }
        if pixels is not None:
            payload["output_activity"] = network.predict_proba(pixels.reshape(1, 784))[0].round(4).tolist()
    payload["input_activity"] = input_activity
    return payload


@app.get("/")
def index():
    """Serve the interactive digit-drawing page."""
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    """Predict a digit from a browser-preprocessed 28 by 28 pixel array."""
    payload = request.get_json(silent=True) or {}
    pixels = np.asarray(payload.get("pixels", []), dtype=np.float32)
    if pixels.size != 784 or not np.all(np.isfinite(pixels)):
        return jsonify(error="Expected exactly 784 finite pixel values."), 400
    pixels = np.clip(pixels, 0, 1)
    active_model = get_model()
    probabilities = active_model.predict_proba(pixels.reshape(1, 784))[0]
    ranking = np.argsort(probabilities)[::-1]
    return jsonify(prediction=int(ranking[0]), confidence=round(float(probabilities[ranking[0]]) * 100, 2), probabilities=[round(float(value) * 100, 2) for value in probabilities], visualization=model_visualization(active_model, pixels))


def run_experiment(epochs: int, learning_rate: float, hidden_size: int) -> None:
    """Train a fresh configuration in a background thread and make it active."""
    global model
    try:
        train_features, train_labels, test_features, test_labels = load_mnist()
        # Keeping experiments responsive while still using genuine MNIST samples.
        train_features, train_labels = train_features[:12000], train_labels[:12000]
        candidate = NeuralNetwork(hidden_size=hidden_size)
        history = []

        def publish_weight_change(epoch: int, batch_number: int) -> None:
            """Publish a snapshot every ten real gradient updates for the UI."""
            if batch_number % 10 == 0:
                training_state["visualization"] = model_visualization(candidate)
                training_state["message"] = f"Epoch {epoch}: updating learned weights (batch {batch_number})."

        for metric in candidate.train(train_features, train_labels, epochs, learning_rate, validation_data=(test_features, test_labels), on_batch_end=publish_weight_change):
            history.append({"epoch": metric.epoch, "loss": round(metric.loss, 4), "accuracy": round(metric.accuracy, 4)})
            training_state["history"] = history
        MODEL_PATH.parent.mkdir(exist_ok=True)
        candidate.save(MODEL_PATH)
        with model_lock:
            model = candidate
        training_state.update(running=False, message=f"Finished. Test accuracy: {history[-1]['accuracy']:.2%}")
    except Exception as error:  # Surface setup/network errors to the browser.
        training_state.update(running=False, message=f"Training failed: {error}")


@app.post("/api/train")
def start_experiment():
    """Start a short MNIST training experiment using the selected hyperparameters."""
    if training_state["running"]:
        return jsonify(error="An experiment is already running."), 409
    payload = request.get_json(silent=True) or {}
    try:
        epochs = int(payload.get("epochs", 3))
        learning_rate = float(payload.get("learning_rate", 0.08))
        hidden_size = int(payload.get("hidden_size", 128))
    except (TypeError, ValueError):
        return jsonify(error="Training settings must be numeric."), 400
    if not 1 <= epochs <= 30 or not 0.001 <= learning_rate <= 1 or not 16 <= hidden_size <= 512:
        return jsonify(error="Use epochs 1–30, learning rate 0.001–1, and hidden layer 16–512."), 400
    training_state.update(running=True, history=[], visualization=None, message="Downloading/loading MNIST and starting training...")
    threading.Thread(target=run_experiment, args=(epochs, learning_rate, hidden_size), daemon=True).start()
    return jsonify(message="Experiment started.")


@app.get("/api/training-status")
def training_status():
    """Return current experiment progress for browser polling."""
    return jsonify(training_state)


@app.get("/api/model-visualization")
def current_model_visualization():
    """Return a weight snapshot for the current saved/active model."""
    return jsonify(model_visualization(get_model()))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
