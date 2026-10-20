# Copyright (c) 2026 Felipe Madison. All rights reserved.
# Licensed under the MIT License.

"""
Linear / Logistic classification and regression model representation
for decentralized distributed training without third-party dependencies.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any


def sigmoid(z: float) -> float:
    # Stable numerical sigmoid
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


class FederatedModel:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.weights: List[float] = [0.0] * dimension
        self.bias: float = 0.0

    def predict_proba(self, x: List[float]) -> float:
        if len(x) != self.dimension:
            raise ValueError(f"Feature length {len(x)} does not match dimension {self.dimension}")
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x: List[float], threshold: float = 0.5) -> int:
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_gradients(
        self, X: List[List[float]], y: List[int]
    ) -> tuple[List[float], float, float]:
        """
        Computes binary cross-entropy loss and parameter gradients over local dataset.
        Returns: (weight_gradients, bias_gradient, total_loss)
        """
        n = len(X)
        if n == 0:
            return ([0.0] * self.dimension, 0.0, 0.0)

        weight_grads = [0.0] * self.dimension
        bias_grad = 0.0
        total_loss = 0.0

        for xi, yi in zip(X, y):
            p = self.predict_proba(xi)
            p_clamped = max(min(p, 1.0 - 1e-12), 1e-12)
            loss_i = -(yi * math.log(p_clamped) + (1 - yi) * math.log(1.0 - p_clamped))
            total_loss += loss_i

            error = p - yi
            for j in range(self.dimension):
                weight_grads[j] += error * xi[j]
            bias_grad += error

        avg_loss = total_loss / n
        weight_grads = [g / n for g in weight_grads]
        bias_grad = bias_grad / n

        return weight_grads, bias_grad, avg_loss

    def apply_gradients(self, weight_grads: List[float], bias_grad: float, lr: float = 0.05) -> None:
        for j in range(self.dimension):
            self.weights[j] -= lr * weight_grads[j]
        self.bias -= lr * bias_grad

    def get_flattened_parameters(self) -> List[float]:
        return self.weights + [self.bias]

    def set_flattened_parameters(self, params: List[float]) -> None:
        if len(params) != self.dimension + 1:
            raise ValueError("Parameter vector size mismatch")
        self.weights = list(params[:self.dimension])
        self.bias = params[self.dimension]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FederatedModel:
        model = cls(dimension=data["dimension"])
        model.weights = list(data["weights"])
        model.bias = float(data["bias"])
        return model
