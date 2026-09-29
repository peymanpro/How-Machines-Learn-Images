from __future__ import annotations

from src.data.synthetic import generate_shape_dataset, train_test_split
from src.learning.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU
from src.learning.losses import classification_accuracy
from src.learning.model import Sequential
from src.learning.optim import SGD, train


def main() -> None:
    inputs, labels = generate_shape_dataset(samples=100, size=20, noise=0.05, seed=17)
    train_x, test_x, train_y, test_y = train_test_split(inputs, labels, seed=19)

    model = Sequential(
        [
            Conv2D(1, 6, kernel_size=3, padding=1, seed=3),
            ReLU(),
            MaxPool2D(),
            Conv2D(6, 8, kernel_size=3, padding=1, seed=4),
            ReLU(),
            MaxPool2D(),
            Flatten(),
            Dense(8 * 5 * 5, 2, seed=5),
        ]
    )
    history = train(model, train_x, train_y, SGD(model.parameters(), 0.03), epochs=25)
    test_logits = model.forward(test_x)

    print(f"final train loss: {history.loss[-1]:.6f}")
    print(f"final train accuracy: {history.accuracy[-1]:.3f}")
    print(f"test accuracy: {classification_accuracy(test_logits, test_y):.3f}")


if __name__ == "__main__":
    main()
