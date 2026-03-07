"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork
from sklearn.metrics import precision_score, recall_score, f1_score

# def parse_arguments():
#     """
#     Parse command-line arguments for inference.
    
#     TODO: Implement argparse with:
#     - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
#     - dataset: Dataset to evaluate on
#     - batch_size: Batch size for inference
#     - hidden_layers: List of hidden layer sizes
#     - num_neurons: Number of neurons in hidden layers
#     - activation: Activation function ('relu', 'sigmoid', 'tanh')
#     """
#     parser = argparse.ArgumentParser(description='Run inference on test set')
    
#     parser.add_argument("-m", "--model_path", type=str)
#     parser.add_argument("-d", "--dataset", type=str, default="mnist")
#     parser.add_argument("-b", "--batch_size", type=int, default=64)
#     parser.add_argument("-nhl", "--num_layers", type=int)
#     parser.add_argument("-sz", "--hidden_size", type=int, nargs="+")
#     parser.add_argument("-a", "--activation", type=str, nargs="+")
    
#     return parser.parse_args()

def parse_arguments():
    """
    Parse command-line arguments for inference
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop")
    parser.add_argument("-nhl", "--num_layers", type=int)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+")
    parser.add_argument("-a", "--activation", type=str, nargs="+")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_p", "--wandb_project", type=str, default="da6401_assignment_1")
    parser.add_argument("-msp", "--model_save_path", type=str)
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data



def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    # Forward pass
    logits = model.forward(X_test)

    # Predictions
    y_pred = np.argmax(logits, axis=1)

    # Loss
    loss = model.compute_loss(logits, y_test)

    # Accuracy
    accuracy = np.mean(y_pred == y_test)

    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    

def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    # Load dataset
    _, _, X_test, y_test = load_dataset(args.dataset)

    # Add input and output sizes
    args.input_size = X_test.shape[1]
    args.output_size = len(np.unique(y_test))

    # Initialize model architecture
    model = NeuralNetwork(args)

    # Load saved weights
    params = load_model(args.model_save_path)

    # Set weights into model
    model.set_weights(params)

    # Evaluate model
    results = evaluate_model(model, X_test, y_test)

    print("Evaluation complete!")

    return results


if __name__ == '__main__':
    main()
