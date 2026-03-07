"""
Creative Failure Visualizations for MNIST Best Model
Plots:
  1. Standard Confusion Matrix
  2. Failure Gallery (misclassified images)
  3. Misclassification Network Graph
  4. Confidence vs Correctness Scatter
  5. t-SNE of Misclassified Samples
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import argparse

from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(model_path):
    npz = np.load(model_path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def get_softmax_probs(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Plot 1: Standard Confusion Matrix ─────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, acc):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(
        f"Confusion Matrix — 10% Validation Split\n"
        f"3-layer | 128 neurons | ReLU | RMSProp | lr=0.001 | bs=128 | Xavier | MSE\n"
        f"Accuracy: {acc:.4f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("plot1_confusion_matrix.png", dpi=150)
    print("Saved: plot1_confusion_matrix.png")
    plt.show()


# ── Plot 2: Failure Gallery ────────────────────────────────────────────────────

def plot_failure_gallery(X_val, y_true, y_pred, probs, n_per_pair=2):
    wrong_idx = np.where(y_true != y_pred)[0]

    error_pairs = {}
    for i in wrong_idx:
        pair = (y_true[i], y_pred[i])
        error_pairs[pair] = error_pairs.get(pair, []) + [i]

    top_pairs = sorted(error_pairs, key=lambda p: len(error_pairs[p]), reverse=True)[:12]

    fig, axes = plt.subplots(len(top_pairs), n_per_pair, figsize=(n_per_pair * 2, len(top_pairs) * 2))
    fig.suptitle("Failure Gallery — Top Misclassified Pairs\n(True → Predicted)", fontsize=13, fontweight='bold')

    for row, pair in enumerate(top_pairs):
        true_lbl, pred_lbl = pair
        samples = error_pairs[pair][:n_per_pair]
        for col in range(n_per_pair):
            ax = axes[row][col]
            if col < len(samples):
                idx = samples[col]
                img = X_val[idx].reshape(28, 28)
                conf = probs[idx][pred_lbl]
                ax.imshow(img, cmap='gray')
                ax.set_title(f"True:{true_lbl} → Pred:{pred_lbl}\nConf:{conf:.2f}", fontsize=7)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("plot2_failure_gallery.png", dpi=150)
    print("Saved: plot2_failure_gallery.png")
    plt.show()


# ── Plot 3: Misclassification Network Graph ────────────────────────────────────

def plot_misclassification_network(y_true, y_pred):
    cm_matrix = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm_matrix, 0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Misclassification Network\n(Edge thickness = confusion count)", fontsize=13, fontweight='bold')

    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    positions = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(10)}

    max_errors = cm_matrix.max()
    cmap = plt.cm.Reds

    for i in range(10):
        for j in range(10):
            if i != j and cm_matrix[i, j] > 0:
                x_start, y_start = positions[i]
                x_end, y_end = positions[j]
                weight = cm_matrix[i, j] / max_errors
                ax.annotate("",
                    xy=(x_end * 0.85, y_end * 0.85),
                    xytext=(x_start * 0.85, y_start * 0.85),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.2,head_length=0.1",
                        color=cmap(weight),
                        lw=weight * 4 + 0.3,
                        alpha=0.7
                    )
                )

    for i, (x, y) in positions.items():
        total_errors = cm_matrix[i].sum() + cm_matrix[:, i].sum()
        circle = plt.Circle((x, y), 0.12, color='steelblue', zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha='center', va='center', fontsize=14,
                fontweight='bold', color='white', zorder=6)
        ax.text(x * 1.22, y * 1.22, f"err:{total_errors}", ha='center',
                va='center', fontsize=7, color='gray')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_errors))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label='Confusion Count')

    plt.tight_layout()
    plt.savefig("plot3_misclassification_network.png", dpi=150)
    print("Saved: plot3_misclassification_network.png")
    plt.show()


# ── Plot 4: Confidence vs Correctness ─────────────────────────────────────────

def plot_confidence_vs_correctness(y_true, y_pred, probs):
    confidence = np.max(probs, axis=1)
    correct = (y_true == y_pred).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confidence vs Correctness", fontsize=13, fontweight='bold')

    ax = axes[0]
    colors = ['#e74c3c' if c == 0 else '#2ecc71' for c in correct]
    ax.scatter(range(len(confidence)), confidence, c=colors, alpha=0.3, s=5)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Max Softmax Confidence")
    ax.set_title("Per-Sample Confidence\n(Green=Correct, Red=Wrong)")
    green_patch = mpatches.Patch(color='#2ecc71', label='Correct')
    red_patch   = mpatches.Patch(color='#e74c3c', label='Wrong')
    ax.legend(handles=[green_patch, red_patch])

    ax2 = axes[1]
    ax2.hist(confidence[correct == 1], bins=40, alpha=0.6, color='#2ecc71', label='Correct')
    ax2.hist(confidence[correct == 0], bins=40, alpha=0.6, color='#e74c3c', label='Wrong')
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution\nCorrect vs Wrong Predictions")
    ax2.legend()

    confidently_wrong = np.sum((correct == 0) & (confidence > 0.9))
    ax2.text(0.05, 0.95, f"Confidently wrong\n(conf>0.9): {confidently_wrong}",
             transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig("plot4_confidence_vs_correctness.png", dpi=150)
    print("Saved: plot4_confidence_vs_correctness.png")
    plt.show()


# ── Plot 5: t-SNE of Misclassified Samples ────────────────────────────────────

def plot_tsne_failures(X_val, y_true, y_pred, max_samples=1000):
    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) > max_samples:
        wrong_idx = np.random.choice(wrong_idx, max_samples, replace=False)

    X_wrong  = X_val[wrong_idx]
    y_wrong  = y_true[wrong_idx]
    yp_wrong = y_pred[wrong_idx]

    print(f"Running t-SNE on {len(wrong_idx)} misclassified samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    X_2d = tsne.fit_transform(X_wrong)

    cmap = plt.cm.get_cmap('tab10', 10)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("t-SNE of Misclassified Samples", fontsize=13, fontweight='bold')

    for ax, labels, title in zip(axes,
                                  [y_wrong, yp_wrong],
                                  ["Colored by True Label", "Colored by Predicted Label"]):
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap,
                        vmin=0, vmax=9, alpha=0.7, s=15)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(sc, ax=ax, ticks=range(10), label='Digit')

    plt.tight_layout()
    plt.savefig("plot5_tsne_failures.png", dpi=150)
    print("Saved: plot5_tsne_failures.png")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",   "--dataset",         type=str,   default="mnist")
    parser.add_argument("-msp", "--model_save_path", type=str,   required=True)
    parser.add_argument("-nhl", "--num_layers",      type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",     type=int,   nargs="+", default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",      type=str,   nargs="+", default=["relu", "relu", "relu"])
    parser.add_argument("-o",   "--optimizer",       type=str,   default="rmsprop")
    parser.add_argument("-lr",  "--learning_rate",   type=float, default=0.001)
    parser.add_argument("-b",   "--batch_size",      type=int,   default=128)
    parser.add_argument("-e",   "--epochs",          type=int,   default=10)
    parser.add_argument("-l",   "--loss",            type=str,   default="mse")
    parser.add_argument("-w_i", "--weight_init",     type=str,   default="xavier")
    parser.add_argument("-wd",  "--weight_decay",    type=float, default=0.0)
    args = parser.parse_args()

    # ── Load data & split ──────────────────────────────────────────────────────
    X_train, y_train, _, _ = load_dataset(args.dataset)
    _, X_val, _, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"Validation samples: {X_val.shape[0]}")

    # ── Load model ─────────────────────────────────────────────────────────────
    args.input_size  = X_train.shape[1]
    args.output_size = len(np.unique(y_train))
    model = NeuralNetwork(args)
    model.set_weights(load_model(args.model_save_path))

    # ── Inference ──────────────────────────────────────────────────────────────
    logits = model.forward(X_val)
    probs  = get_softmax_probs(logits)
    y_pred = np.argmax(probs, axis=1)
    acc    = np.mean(y_pred == y_val)
    print(f"Accuracy: {acc:.4f}")

    # ── All Plots ──────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_val, y_pred, acc)
    plot_failure_gallery(X_val, y_val, y_pred, probs)
    plot_misclassification_network(y_val, y_pred)
    plot_confidence_vs_correctness(y_val, y_pred, probs)
    plot_tsne_failures(X_val, y_val, y_pred)


if __name__ == "__main__":
    main()