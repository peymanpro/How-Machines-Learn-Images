from __future__ import annotations

import argparse

from src.data.mnist import load_mnist
from src.learning.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU
from src.learning.losses import classification_accuracy
from src.learning.model import Sequential
from src.learning.optim import SGD, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NumPy CNN on MNIST IDX files.")
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--test-images", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    train_images, train_labels = load_mnist(args.train_images, args.train_labels)
    test_images, test_labels = load_mnist(args.test_images, args.test_labels)
    train_images = train_images[: args.train_limit]
    train_labels = train_labels[: args.train_limit]
    test_images = test_images[: args.test_limit]
    test_labels = test_labels[: args.test_limit]

    model = Sequential(
        [
            Conv2D(1, 4, kernel_size=3, padding=1, seed=6),
            ReLU(),
            MaxPool2D(),
            Flatten(),
            Dense(4 * 14 * 14, 10, seed=7),
        ]
    )
    history = train(
        model,
        train_images,
        train_labels,
        SGD(model.parameters(), args.learning_rate),
        epochs=args.epochs,
    )
    accuracy = classification_accuracy(model.forward(test_images), test_labels)

    print(f"final loss: {history.loss[-1]:.6f}")
    print(f"final train accuracy: {history.accuracy[-1]:.3f}")
    print(f"test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
