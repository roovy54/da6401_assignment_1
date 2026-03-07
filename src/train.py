"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import os 
import numpy as np
import json
import wandb
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments.
    
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
    parser = argparse.ArgumentParser(description='Train a neural network')
    
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
    parser.add_argument("-w_x", "--experiment_name", type=str)
    parser.add_argument("-msp", "--model_save_path", type=str, default="models/")

    return parser.parse_args()


def main():
    """
    Main training function.
    """

    # parse command-line arguments
    args = parse_arguments()

    if args.experiment_name is None:
        args.experiment_name = (
            f"{args.optimizer}_"
            f"lr{args.learning_rate}_"
            f"bs{args.batch_size}_"
            f"ep{args.epochs}_"
            f"nl{args.num_layers}_"
            f"hs{'-'.join(map(str, args.hidden_size))}_"
            f"act{'-'.join(args.activation)}_"
            f"loss{args.loss}_"
            f"init{args.weight_init}_"
            f"wd{args.weight_decay}"
        )

    wandb.init(project=args.wandb_project, name=args.experiment_name)

    # config = wandb.config
    # args.optimizer = config.optimizer
    # args.learning_rate = config.learning_rate
    # args.batch_size = config.batch_size
    # args.epochs = config.epochs
    # args.num_layers = config.num_layers
    # args.hidden_size = [config.hidden_size] * args.num_layers  
    # args.activation = [config.activation] * args.num_layers
    # args.loss = config.loss
    # args.weight_init = config.weight_init
    # args.weight_decay = config.weight_decay


    # Load dataset based on the specified argument
    X_train, y_train, _, _ = load_dataset(args.dataset)

    # Add input_size and output_size dynamically
    args.input_size = X_train.shape[1]          # 784 if flattened
    args.output_size = len(np.unique(y_train))  # 10 classes

    # Initialize and train the neural network
    model = NeuralNetwork(args)
    model.train(X_train, y_train, args)

    # # Evaluate the model on the test set
    # accuracy = model.evaluate(X_test, y_test)
    # print("Model Evaluation Accuracy:", accuracy)

    # save the trained model weights to disk
    save_dir = args.model_save_path
    os.makedirs(save_dir, exist_ok=True)

    filename = f"model_{args.dataset}_nl{args.num_layers}_hs{'-'.join(map(str, args.hidden_size))}_act{'-'.join(args.activation)}_opt{args.optimizer}_lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}.npz"
    save_path = os.path.join(save_dir, filename)

    # Create dictionary of parameters
    params = model.get_weights()

    # Save everything into one file
    np.savez(save_path, **params)

    print(f"Model saved at {save_path}!")

    # Build config dictionary
    best_config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "loss": args.loss,
        "weight_init": args.weight_init,
        "weight_decay": args.weight_decay,
        "input_size": args.input_size,
        "output_size": args.output_size,
        "wandb_project": args.wandb_project
    }

    # Save JSON file
    config_path = os.path.join(save_dir, "best_config.json")

    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)

    print(f"Configuration saved at {config_path}")
    print("Training complete!")

    wandb.finish()


if __name__ == '__main__':
    main()
