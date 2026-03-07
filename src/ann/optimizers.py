"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class Optimizer:
    def __init__(self, learning_rate, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def step(self, layers):
        raise NotImplementedError

class SGD(Optimizer):
    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):

                if self.weight_decay > 0:
                    layer.grad_W += self.weight_decay * layer.W

                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b

class Momentum(Optimizer):
    def __init__(self, learning_rate, beta=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.velocities = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue

            if i not in self.velocities:
                self.velocities[i] = {
                    "vW": np.zeros_like(layer.W),
                    "vB": np.zeros_like(layer.b)
                }

            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            vW = self.velocities[i]["vW"]
            vB = self.velocities[i]["vB"]

            vW = self.beta * vW - self.lr * layer.grad_W
            vB = self.beta * vB - self.lr * layer.grad_b

            layer.W += vW
            layer.b += vB

            self.velocities[i]["vW"] = vW
            self.velocities[i]["vB"] = vB

class NAG(Optimizer):
    def __init__(self, learning_rate, beta=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.velocities = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue

            if i not in self.velocities:
                self.velocities[i] = {
                    "vW": np.zeros_like(layer.W),
                    "vB": np.zeros_like(layer.b)
                }

            # L2 Regularization
            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            vW_prev = self.velocities[i]["vW"]
            vB_prev = self.velocities[i]["vB"]

            # Update velocity
            vW = self.beta * vW_prev - self.lr * layer.grad_W
            vB = self.beta * vB_prev - self.lr * layer.grad_b

            # Nesterov update
            layer.W += -self.beta * vW_prev + (1 + self.beta) * vW
            layer.b += -self.beta * vB_prev + (1 + self.beta) * vB

            self.velocities[i]["vW"] = vW
            self.velocities[i]["vB"] = vB

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9,
                 eps=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.eps = eps
        self.v = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue

            if i not in self.v:
                self.v[i] = {
                    "vW": np.zeros_like(layer.W),
                    "vB": np.zeros_like(layer.b)
                }

            # L2 Regularization
            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            # Update running average of squared gradients
            self.v[i]["vW"] = self.beta * self.v[i]["vW"] + (1 - self.beta) * (layer.grad_W ** 2)
            self.v[i]["vB"] = self.beta * self.v[i]["vB"] + (1 - self.beta) * (layer.grad_b ** 2)
            # Update parameters
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.v[i]["vW"]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.v[i]["vB"]) + self.eps)
