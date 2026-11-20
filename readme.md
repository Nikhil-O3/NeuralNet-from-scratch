# NumPy MNIST Digit Lab

A Flask app that trains a handwritten-digit classifier **from scratch with NumPy**. It implements the forward pass, ReLU, softmax, cross-entropy loss, backpropagation, and mini-batch gradient descent itself—there is no TensorFlow, PyTorch, Keras, or pretrained model.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py --epochs 15 --learning-rate 0.08 --hidden-size 128
.venv/bin/python app.py
```

Open `http://127.0.0.1:5000`, draw a large single digit, and select **Predict digit**. The first training run downloads the public original MNIST IDX files to `data/`; the trained weight arrays are saved as `models/mnist_numpy_model.npz`.

The web app can also run shorter experiments with learning rate, epochs, and hidden-layer size. Each experiment creates and saves a newly trained model from MNIST samples, then uses that model for predictions.

## Verify

```bash
.venv/bin/python -m pytest -q
```

The checked-in `models/mnist_numpy_model.npz` was trained locally from random weights on all 60,000 MNIST training images. Its final accuracy on the separate 10,000-image test set was **97.21%**.

## Project layout

- `neural_network.py` — manual NumPy neural-network math.
- `mnist_data.py` — download and IDX parsing with the Python standard library.
- `train.py` — reproducible full training command.
- `app.py` — Flask routes for prediction and experiments.
- `templates/` and `static/` — canvas drawing interface and preprocessing.
