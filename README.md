# Assignment 1: Multi-Layer Perceptron (MLP) for Image Classification

## Overview

This repository contains an implementation of a Multi-Layer Perceptron (MLP) built from scratch using NumPy. The goal is to learn the fundamentals of forward/backward propagation, build layers/activations/optimizers manually, and train an MLP on image classification datasets such as MNIST or Fashion-MNIST.

Key features
- Custom neural network implementation under `src/ann/` (layers, activations, objectives, optimizers)
- Training and inference scripts (`src/train.py`, `src/inference.py`)
- Example notebooks (training/evaluation and Weights & Biases experiments)
- Example model artifacts in `src/` (e.g., `best_model.npz`, `best_model.npy`) and a best config JSON

## Table of contents

- [Requirements](#requirements)
- [Install](#install)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Usage details](#usage-details)
- [Notebooks & experiments](#notebooks--experiments)
- [Notes, assumptions and next steps](#notes-assumptions-and-next-steps)

## Requirements

This project uses Python 3.x and the Python dependencies listed in `requirements.txt`.

Install requirements (recommended inside a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have the packages installed, ensure versions are compatible by checking `requirements.txt`.

## Install

Clone the repo and install dependencies (if not already done):

```bash
git clone <this-repo-url>
cd da6401_assignment_1
# create & activate virtual env, then:
pip install -r requirements.txt
```

## Quick start

- To train a model (basic):

```bash
python src/train.py
```

- To run inference on saved model files:

```bash
python src/inference.py
```

Notes: The example scripts (`train.py`, `inference.py`) may accept command-line arguments (dataset choice, hyperparameters, paths). See the top of each script or run `python src/train.py --help` / `python src/inference.py --help` for available flags.

## Project structure

Top-level layout

```
da6401_assignment_1/
├── models/                 # (empty placeholder / outputs)
├── notebooks/              # Jupyter notebooks for experiments & WandB demos
├── src/
│   ├── ann/                # core MLP implementation
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
	 │   ├── objective_functions.py
	 │   └── optimizers.py
│   ├── plots/              # (plot utilities / saved figures)
│   ├── utils/
│   │   └── data_loader.py  # dataset loading utilities
│   ├── train.py            # training entrypoint
│   ├── inference.py        # inference / evaluation script
│   ├── best_config.json    # example best hyperparameter config
│   ├── best_model.npy      # example saved model weights (NumPy)
│   └── best_model.npz      # example model checkpoint
└── requirements.txt
```

Brief file descriptions
- `src/ann/activations.py`: activation functions and derivatives (ReLU, sigmoid, softmax, etc.)
- `src/ann/neural_layer.py`: an implementation of a dense/fully-connected layer
- `src/ann/neural_network.py`: builds and orchestrates layers into a network; forward/backward passes
- `src/ann/objective_functions.py`: loss functions (cross-entropy, MSE, etc.)
- `src/ann/optimizers.py`: optimizer implementations (SGD, Momentum, Adam, etc.)
- `src/utils/data_loader.py`: routines to load MNIST/Fashion-MNIST and prepare batches
- `src/train.py`: training loop, logging, saving models/configs
- `src/inference.py`: load a saved model and run predictions / evaluation

Model artifacts
- `src/best_model.npz`, `src/best_model.npy`: example saved model weights/checkpoints
- `src/best_config.json`: hyperparameters used for the saved best model

## Usage details

1) Training

- Basic (defaults):

```bash
python src/train.py
```

- Common flags you may want to change (check `train.py` for exact names):
	- dataset (mnist / fashion-mnist)
	- learning rate, optimizer, batch size, epochs
	- output/model save path

Training produces saved model files (e.g., `.npz` or `.npy`) and a config JSON with the hyperparameters.

2) Inference / Evaluation

- Basic usage:

```bash
python src/inference.py --model src/best_model.npz
```

- The `inference.py` script typically supports options to point to a model file, specify an input image or dataset split, and choose output formats. Check the script header or `--help` for supported options.

3) Visualizing / Notebooks

- `notebooks/wandb_demo.ipynb` and `notebooks/wandb_report.ipynb` contain experiment demos and reports. Open them in JupyterLab / Jupyter Notebook.
- `notebooks/sweep.yaml` is a Weights & Biases sweep configuration for hyperparameter tuning.

## Notebooks & experiments

- To run the notebooks, start Jupyter in the repo root:

```bash
jupyter lab
```

- If you use Weights & Biases (wandb), the notebooks and `train.py` may log runs to your account. Configure your WandB API key before running experiments (see wandb docs).

## Notes, assumptions and next steps

Assumptions made while writing this README:
- The `train.py` and `inference.py` scripts provide CLI flags (or are easy to edit) to point to dataset/model paths. If they require a different interface, inspect those files for exact usage.
- Model files are stored in `src/` for convenience in this repo; in a production workflow you'd place them under `models/`.

Recommended next steps / low-risk improvements:
- Add `--help` output to `train.py` and `inference.py` if missing (argparse)
- Add a small `examples/` directory with example commands and sample inputs for inference
- Add a short CONTRIBUTING.md describing how to run tests and extend the project

## Contact / Support

If this is your course assignment, follow course submission instructions. For issues with the code, open an issue or contact the project owner.

---

Good luck and enjoy experimenting with your MLP!
