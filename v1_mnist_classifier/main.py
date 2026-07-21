"""
main.py
=======

WHAT THIS FILE DOES (in plain English)
---------------------------------------
This is the "conductor" — it doesn't do any of the actual work itself,
but it brings together the data (dataset.py), the model (model.py), the
training loop (train.py), and the evaluation code (evaluate.py) in the
right order, and manages the handful of settings ("hyperparameters")
that control the whole process.

Run this file with:  python main.py

If you're new to all this: read this file top to bottom, and read the
detailed comments in src/dataset.py, src/model.py, src/train.py, and
src/evaluate.py for a full explanation of every concept used here. This
file just calls into those, in order.

------------------------------------------------------------------
BACKGROUND CONCEPT: What are "hyperparameters"?
------------------------------------------------------------------
A "hyperparameter" is a setting YOU choose before training starts, as
opposed to a "parameter" (a weight/bias), which the model itself learns
automatically during training. The constants below (BATCH_SIZE,
LEARNING_RATE, EPOCHS, etc.) are all hyperparameters. There's no single
"correct" value for most of them — they're starting points that work
well in practice for this kind of problem, based on established
convention (reuse-first applies to hyperparameter choices too: these
aren't guesses, they're widely-used defaults for MNIST-scale problems).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import MLPClassifier
from src.train import train
from src.evaluate import evaluate, print_confusion_matrix

# ============================================================
# CONFIG — all the adjustable settings for this run, in one place
# ============================================================

# How many images the model looks at per training step, before updating
# its weights once. Smaller batches = noisier but more frequent updates;
# larger batches = smoother but slower updates. 64 is a common default
# that works well for a dataset and model this size.
BATCH_SIZE = 64

# How many "neurons" (values) live in the model's hidden layer, between
# the two Linear layers (see src/model.py). More = the model can
# represent more complex patterns, but also trains slower and risks
# overfitting on a simple problem like this. 128 is a solid starting point.
HIDDEN_SIZE = 128

# Controls how big each weight-update step is during training (see the
# backpropagation/optimizer explanation in src/train.py). 1e-3 (0.001)
# is the standard, well-tested default for the Adam optimizer across a
# huge range of problems — a reliable starting point, not a guess.
LEARNING_RATE = 1e-3

# How many complete passes over the entire training dataset to do.
# MNIST is a small, "easy" dataset by modern standards, so 5 epochs is
# already enough to get very good (~97-98%) accuracy with this simple
# model. More complex datasets/models typically need many more epochs.
EPOCHS = 5

# Where to store/look for the downloaded MNIST files.
DATA_DIR = "./data"

# Where to save the trained model's weights once training finishes, so
# you could later reload and reuse this exact trained model without
# retraining from scratch.
CHECKPOINT_PATH = "./saved_models/mlp_mnist.pt"


def main():
    # ------------------------------------------------------------
    # STEP 0: Choose the compute device (CPU vs GPU)
    # ------------------------------------------------------------
    # A GPU (graphics card) can do the huge number of matrix
    # multiplications neural networks require MUCH faster than a CPU,
    # but not every machine has one available. This line automatically
    # uses a GPU ("cuda") if PyTorch detects one, and falls back to the
    # regular CPU otherwise — either works fine for a dataset this small,
    # a GPU will just make training noticeably faster.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # STEP 1: Load the data
    # ------------------------------------------------------------
    # See src/dataset.py for the full explanation of what happens here
    # (downloading MNIST, normalizing pixel values, batching).
    train_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

    # ------------------------------------------------------------
    # STEP 2: Build the model
    # ------------------------------------------------------------
    # Creates a fresh MLPClassifier with RANDOMLY initialized weights
    # (see src/model.py for the architecture details), then moves it onto
    # whichever device we picked in Step 0. Right now, before any
    # training, this model's predictions are essentially random guesses.
    model = MLPClassifier(hidden_size=HIDDEN_SIZE).to(device)

    # ------------------------------------------------------------
    # STEP 3: Define how we measure "wrongness" and how we fix it
    # ------------------------------------------------------------
    # CrossEntropyLoss: the standard loss function for classification
    # problems (see src/train.py for the full explanation).
    criterion = nn.CrossEntropyLoss()

    # Adam: the optimizer that will use gradients to update the model's
    # weights during training (see src/train.py for the full explanation).
    # model.parameters() hands the optimizer every single weight/bias in
    # the model so it knows what it's allowed to update.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------
    # STEP 4: Train
    # ------------------------------------------------------------
    # This is where the actual learning happens, over EPOCHS full passes
    # through the training data. Watch the printed train_loss go down and
    # train_acc go up as it progresses. See src/train.py for the full
    # step-by-step explanation of what happens inside each pass.
    train(model, train_loader, optimizer, criterion, device, EPOCHS)

    # ------------------------------------------------------------
    # STEP 5: Evaluate on unseen test data
    # ------------------------------------------------------------
    # Now that training is done, check how well the model performs on
    # images it has NEVER seen before — this is the honest measure of
    # how good the model actually is. See src/evaluate.py for details.
    accuracy, confusion = evaluate(model, test_loader, device)
    print(f"\nTest accuracy: {accuracy:.4f}")
    print_confusion_matrix(confusion)

    # ------------------------------------------------------------
    # STEP 6: Save the trained model to disk
    # ------------------------------------------------------------
    # `model.state_dict()` is a snapshot of every learned weight and bias
    # in the model, as a simple dictionary of tensors. Saving just this
    # (rather than the whole Python object) is the standard, recommended
    # PyTorch way to persist a trained model — it's small, portable, and
    # future versions of PyTorch can still load it, since it doesn't
    # depend on this exact Python class definition being unchanged.
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nModel saved to {CHECKPOINT_PATH}")


# This is a standard Python convention: the code inside this `if` block
# only runs when you execute this file DIRECTLY (`python main.py`), not
# if some other file imports something from main.py. It's good habit
# to always structure entry-point scripts this way.
if __name__ == "__main__":
    main()
