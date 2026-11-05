"""Numerical gradient check for the NumPy CNN (run: python -m pytest tests)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from depresjon_cnn.cnn import ActivityCNN  # noqa: E402


def _loss(model, x, y):
    out = model._forward(x, train=False)
    return model._loss_grad(out, y, np.ones_like(y))[0]


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    model = ActivityCNN(dropout=0.0)
    model._build()
    x = rng.normal(size=(4, 1, 48))
    y = np.array([0.0, 1.0, 1.0, 0.0])

    out = model._forward(x, train=False)
    _, dout = model._loss_grad(out, y, np.ones_like(y))
    model._backward(dout)

    eps = 1e-5
    for layer in model.layers:
        for name, param in layer.params.items():
            flat = param.reshape(-1)
            for i in rng.choice(flat.size, size=min(5, flat.size), replace=False):
                old = flat[i]
                flat[i] = old + eps
                up = _loss(model, x, y)
                flat[i] = old - eps
                down = _loss(model, x, y)
                flat[i] = old
                numeric = (up - down) / (2 * eps)
                analytic = layer.grads[name].reshape(-1)[i]
                assert abs(numeric - analytic) <= 1e-4 + 1e-3 * abs(numeric), (
                    type(layer).__name__, name, numeric, analytic)


if __name__ == "__main__":
    test_gradients_match_finite_differences()
    print("gradient check passed")
