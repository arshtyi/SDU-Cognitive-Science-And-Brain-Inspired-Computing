import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

type Array = NDArray[np.float64]
type Labels = NDArray[np.int64]

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
LEARNING_RATE = 1.0
MAX_EPOCHS = 100
TRUTH_TABLES = {
    "AND": [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]],
    "OR": [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
    "NOT": [[0, 1], [1, 0]],
}


def predict(inputs: Array, weights: Array, bias: float) -> Labels:
    return (inputs @ weights + bias >= 0).astype(np.int64)


def train(
    inputs: Array,
    targets: Labels,
    learning_rate: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[Array, float, list[int]]:
    weights = np.zeros(inputs.shape[1], dtype=np.float64)
    bias = 0.0
    mistakes = []
    for _ in range(max_epochs):
        count = 0
        for sample, target in zip(inputs, targets, strict=True):
            prediction = predict(sample[None, :], weights, bias)[0]
            error = int(target - prediction)
            weights += learning_rate * error * sample
            bias += learning_rate * error
            count += int(error != 0)
        mistakes.append(count)
        if count == 0:
            return weights, bias, mistakes
    raise RuntimeError("在最大训练轮数内未收敛")


def plot_history(histories: dict[str, list[int]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    for name, mistakes in histories.items():
        ax.plot(
            range(1, len(mistakes) + 1),
            mistakes,
            "o-",
            label=name,
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Online mistakes / updates",
        xticks=range(1, max(map(len, histories.values())) + 1),
        yticks=range(5),
        ylim=(-0.2, 4.2),
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(OUTPUT / "training.svg", metadata={"Date": None})
    plt.close(fig)


OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "perceptron-lab2"})
histories = {}
with (
    (OUTPUT / "results.csv").open("w", encoding="utf-8", newline="") as results_file,
    (OUTPUT / "models.csv").open("w", encoding="utf-8", newline="") as models_file,
):
    results = csv.writer(results_file)
    models = csv.writer(models_file)
    results.writerow(["gate", "inputs", "target", "net", "prediction"])
    models.writerow(["gate", "weights", "bias", "epochs", "correct", "total"])
    for name, rows in TRUTH_TABLES.items():
        data = np.array(rows, dtype=np.float64)
        inputs = data[:, :-1]
        targets = data[:, -1].astype(np.int64)
        weights, bias, mistakes = train(inputs, targets)
        predictions = predict(inputs, weights, bias)
        net_inputs = inputs @ weights + bias
        correct = int(np.count_nonzero(predictions == targets))
        histories[name] = mistakes
        models.writerow([name, weights.tolist(), bias, len(mistakes), correct, len(rows)])
        print(f"\n{name}: w={weights}, b={bias:g}")
        print(f"训练 {len(mistakes)} 轮，正确 {correct}/{len(rows)}")
        for row, net, prediction in zip(rows, net_inputs, predictions, strict=True):
            sample, target = row[:-1], row[-1]
            results.writerow([name, sample, target, net, prediction])
            print(f"  {sample} -> {prediction}（期望 {target}，净输入 {net:g}）")
plot_history(histories)
