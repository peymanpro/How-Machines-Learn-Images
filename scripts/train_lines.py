from __future__ import annotations

from src.data.synthetic import generate_line_dataset, train_test_split
from src.learning.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU
from src.learning.losses import classification_accuracy
from src.learning.model import Sequential
from src.learning.optim import SGD, train


def main() -> None:
    inputs, labels = generate_line_dataset(samples=80, size=16, noise=0.04, seed=7)
    train_x, test_x, train_y, test_y = train_test_split(inputs, labels, seed=11)

    model = Sequential(
        [
            Conv2D(1, 4, kernel_size=3, padding=1, seed=1),
            ReLU(),
            MaxPool2D(),
            Flatten(),
            Dense(4 * 8 * 8, 2, seed=2),
        ]
    )
    history = train(model, train_x, train_y, SGD(model.parameters(), 0.05), epochs=20)
    test_logits = model.forward(test_x)

    print(f"final train loss: {history.loss[-1]:.6f}")
    print(f"final train accuracy: {history.accuracy[-1]:.3f}")
    print(f"test accuracy: {classification_accuracy(test_logits, test_y):.3f}")


if __name__ == "__main__":
    main()
