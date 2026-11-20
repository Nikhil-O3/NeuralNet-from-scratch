import numpy as np

from neural_network import NeuralNetwork


def test_training_learns_a_simple_non_linear_problem(tmp_path):
    """Backpropagation should learn XOR, which a linear model cannot solve."""
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    labels = np.array([0, 1, 1, 0])
    model = NeuralNetwork(input_size=2, hidden_size=8, output_size=2, seed=2)
    list(model.train(features, labels, epochs=500, learning_rate=0.1, batch_size=4))
    assert np.array_equal(np.argmax(model.predict_proba(features), axis=1), labels)
    model_path = tmp_path / "xor.npz"
    model.save(model_path)
    assert np.allclose(NeuralNetwork.load(model_path).predict_proba(features), model.predict_proba(features))
