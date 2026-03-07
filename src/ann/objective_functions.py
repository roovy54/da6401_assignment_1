"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from .activations import softmax

class CrossEntropyLoss:
    """
    Cross-Entropy Loss for multi-class classification.
    L = -1/m * sum(log(p_i)) where p_i is the predicted probability of the true class for each sample.
    """
    
    def forward(self, y_true, logits):
        m = y_true.shape[0]
        probs = softmax(logits)
        
        log_likelihood = -np.log(probs[np.arange(m), y_true])
        loss = np.sum(log_likelihood) / m
        
        return loss
    
    def backward(self, y_true, logits):
        m = y_true.shape[0]
        probs = softmax(logits)
        
        probs[np.arange(m), y_true] -= 1
        dZ = probs / m
        
        return dZ


class MSELoss:
    """
    Mean Squared Error Loss for regression tasks. 
    Here for comparing with cross-entropy in classification.
    L = 1/m * sum((y_pred - y_true)^2)
    """

    def forward(self, y_true, y_pred):
        m = y_pred.shape[0]

        # Convert class indices to one-hot if needed
        if y_true.ndim == 1:
            num_classes = y_pred.shape[1]
            y_true = np.eye(num_classes)[y_true]

        loss = np.sum((y_pred - y_true) ** 2) / m
        return loss
    

    def backward(self, y_true, y_pred):
        m = y_pred.shape[0]

        # Convert class indices to one-hot if needed
        if y_true.ndim == 1:
            num_classes = y_pred.shape[1]
            y_true = np.eye(num_classes)[y_true]

        dZ = (2 / m) * (y_pred - y_true)
        return dZ