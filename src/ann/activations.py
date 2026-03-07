"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

def relu(x):
    """
    ReLU activation function
    f(x) = max(0, x)
    Note: Introduces non-linearity and helps with vanishing gradient problem.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU activation function
    f'(x) = 1 if x > 0 else 0
    Note: We assume the derivative at x=0 is 0 for simplicity.
    """
    return np.where(x > 0, 1, 0) 

def sigmoid(x):
    """
    Sigmoid activation function
    f(x) = 1 / (1 + e^(-x))
    Note: Maps any real number to a value between 0 and 1. Mainly used in output layer for binary classification. Can cause vanishing gradient problem for deep networks.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of Sigmoid activation function
    f'(x) = f(x) * (1 - f(x))
    Note: Expressed in terms of the sigmoid function itself, making it computationally efficient. 
    """
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """
    Tanh activation function
    f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Note: Maps any real number to a value between -1 and 1. Zero-centered, which can help with convergence. Can also cause vanishing gradient problem
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of Tanh activation function
    f'(x) = 1 - f(x)^2
    Note: Expressed in terms of the tanh function itself, making it computationally efficient
    """
    t = tanh(x)
    return 1 - t**2

def softmax(x):
    """
    Softmax activation function
    f(x_i) = e^x_i / sum(e^x_j for all j)
    Note: Used in output layer for multi-class classification problems. Ensures output values sum to 1.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)       