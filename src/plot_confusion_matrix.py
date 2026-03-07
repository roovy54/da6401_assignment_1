"""
Plot Confusion Matrix on 10% validation split of X_train
Uses the project's NeuralNetwork and load_dataset utilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork


def load_model(model_path):
    npz = np.load(model_path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist")
    parser.add_argument("-msp", "--model_save_path", type=str, required=True)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",    type=str,   nargs="+", default=["relu", "relu", "relu"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop")
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=128)
    parser.add_argument("-e",   "--epochs",        type=int,   default=10)
    parser.add_argument("-l",   "--loss",          type=str,   default="mse")
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier")
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0)
    args = parser.parse_args()

    # ── 1. Load dataset and take 10% of X_train ───────────────────────────────
    X_train, y_train, _, _ = load_dataset(args.dataset)

    _, X_val, _, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"Validation split size: {X_val.shape[0]} samples")

    # ── 2. Build model and load weights ───────────────────────────────────────
    args.input_size  = X_train.shape[1]
    args.output_size = len(np.unique(y_train))

    model = NeuralNetwork(args)
    params = load_model(args.model_save_path)
    model.set_weights(params)

    # ── 3. Predict ────────────────────────────────────────────────────────────
    logits = model.forward(X_val)
    y_pred = np.argmax(logits, axis=1)

    acc = np.mean(y_pred == y_val)
    print(f"Validation Accuracy (10% split): {acc:.4f}")

    # ── 4. Plot Confusion Matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(args.output_size))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(
        f"Confusion Matrix — 10% Train Split\n"
        f"3-layer | 128 neurons | ReLU | RMSProp | lr=0.001 | bs=128 | Xavier | MSE\n"
        f"Accuracy: {acc:.4f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()