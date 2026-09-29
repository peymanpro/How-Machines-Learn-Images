from __future__ import annotations

import json

from src.data.synthetic import generate_line_dataset
from src.learning.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU
from src.learning.model import Sequential
from src.learning.optim import SGD, train


def build_model(seed_offset: int = 0) -> Sequential:
    return Sequential(
        [
            Conv2D(1, 2, kernel_size=3, padding=1, seed=1 + seed_offset),
            ReLU(),
            MaxPool2D(),
            Flatten(),
            Dense(2 * 4 * 4, 2, seed=2 + seed_offset),
        ]
    )


def main() -> None:
    inputs, labels = generate_line_dataset(samples=40, size=8, noise=0.03, seed=31)
    results = []
    for learning_rate in (0.01, 0.03, 0.05):
        model = build_model(seed_offset=int(learning_rate * 100))
        history = train(
            model,
            inputs,
            labels,
            SGD(model.parameters(), learning_rate),
            epochs=10,
        )
        results.append(
            {
                "learning_rate": learning_rate,
                "initial_loss": history.loss[0],
                "final_loss": history.loss[-1],
                "final_accuracy": history.accuracy[-1],
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
