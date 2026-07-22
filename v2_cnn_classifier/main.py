"""
main.py
=======

WHAT THIS FILE DOES (in plain English)
---------------------------------------
Same conductor role as v1's main.py: ties together data, model, training,
and evaluation, and holds the run's hyperparameters in one place.

The ONLY functional difference from v1: this imports `CNNClassifier`
instead of `MLPClassifier`. Everything else — the training loop, the
evaluation code, the data loading — is byte-for-byte identical to v1.
That's the point of this version: prove that swapping architectures only
requires touching src/model.py, exactly as v1's README promised.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import CNNClassifier
from src.train import train
from src.evaluate import evaluate, print_confusion_matrix

# ============================================================
# CONFIG — all the adjustable settings for this run, in one place
# ============================================================

BATCH_SIZE = 64

# CNNs typically need fewer epochs than an MLP to reach strong accuracy
# on MNIST, since convolution already encodes useful assumptions about
# image structure. 5 epochs (same as v1) is kept for a fair, direct
# comparison between the two versions.
LEARNING_RATE = 1e-3
EPOCHS = 5

DATA_DIR = "./data"
CHECKPOINT_PATH = "./saved_models/cnn_mnist.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

    # Only line that differs from v1: CNNClassifier() instead of
    # MLPClassifier(hidden_size=HIDDEN_SIZE). No hidden_size argument
    # needed here — the CNN's channel counts are fixed inside model.py
    # for this version, since tuning them isn't today's bottleneck.
    model = CNNClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, optimizer, criterion, device, EPOCHS)

    accuracy, confusion = evaluate(model, test_loader, device)
    print(f"\nTest accuracy: {accuracy:.4f}")
    print_confusion_matrix(confusion)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nModel saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
