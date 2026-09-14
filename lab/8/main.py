import csv
from pathlib import Path
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

type Array = NDArray[np.float64]
type Parameters = dict[str, Array]

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SEED = 42
LEARNING_RATE = 0.5
MAX_EPOCHS = 1000
TOLERANCE = 0.01


def initialize() -> Parameters:
    rng = np.random.default_rng(SEED)
    return {
        "W1": rng.uniform(0, 1, (2, 2)),
        "b1": np.zeros((2, 1)),
        "W2": rng.uniform(0, 1, (1, 2)),
        "b2": np.zeros((1, 1)),
    }


def sigmoid(values: Array) -> Array:
    return 0.5 * (1 + np.tanh(values / 2))


def forward(inputs: Array, params: Parameters) -> tuple[Array, Array]:
    hidden = sigmoid(params["W1"] @ inputs + params["b1"])
    prediction = sigmoid(params["W2"] @ hidden + params["b2"])
    return hidden, prediction


def squared_error(prediction: Array, targets: Array) -> float:
    return float(np.sum((prediction - targets) ** 2))


def backward(inputs: Array, targets: Array, hidden: Array, prediction: Array, params: Parameters) -> Parameters:
    delta2 = 2 * (prediction - targets) * prediction * (1 - prediction)
    delta1 = (params["W2"].T @ delta2) * hidden * (1 - hidden)
    return {
        "W1": delta1 @ inputs.T,
        "b1": delta1,
        "W2": delta2 @ hidden.T,
        "b2": delta2,
    }


def check_gradients(inputs: Array, targets: Array, params: Parameters) -> None:
    hidden, prediction = forward(inputs, params)
    gradients = backward(inputs, targets, hidden, prediction, params)
    epsilon = 1e-6
    rows = []
    for name, values in params.items():
        for row, column in np.ndindex(values.shape):
            value = float(values[row, column])
            values[row, column] = value + epsilon
            plus = squared_error(forward(inputs, params)[1], targets)
            values[row, column] = value - epsilon
            minus = squared_error(forward(inputs, params)[1], targets)
            values[row, column] = value
            numerical = (plus - minus) / (2 * epsilon)
            gradient = float(gradients[name][row, column])
            change = -LEARNING_RATE * gradient
            error = abs(gradient - numerical)
            rows.append((name, row, column, value, gradient, numerical, change, value + change, error))
            if not np.isfinite(gradient) or not np.isclose(gradient, numerical, rtol=1e-5, atol=1e-8):
                raise RuntimeError(f"梯度检查失败：{name}[{row}, {column}]")
    with (OUTPUT / "first_step.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["parameter", "row", "column", "initial", "gradient", "numerical_gradient", "delta", "updated", "absolute_error"])
        writer.writerows(rows)
    print(f"全部 {len(rows)} 个参数的梯度检查通过；最大绝对误差 {max(row[-1] for row in rows):.3e}")


def train(inputs: Array, targets: Array, params: Parameters) -> Array:
    history = []
    for epoch in range(MAX_EPOCHS + 1):
        hidden, prediction = forward(inputs, params)
        loss = squared_error(prediction, targets)
        history.append((epoch, prediction.item(), loss))
        print(f"Update {epoch:4d}: output={prediction.item():.9f}, SSE={loss:.9f}")
        if not np.isfinite(loss):
            raise RuntimeError("训练误差不是有限数值")
        if loss < TOLERANCE:
            return np.array(history)
        if epoch < MAX_EPOCHS:
            gradients = backward(inputs, targets, hidden, prediction, params)
            for name, values in params.items():
                values -= LEARNING_RATE * gradients[name]
    raise RuntimeError(f"完成 {MAX_EPOCHS} 次更新后，平方和误差仍未小于 {TOLERANCE}")


def plot_history(history: Array) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    axes[0].plot(history[:, 0], history[:, 1], "o-", label="Prediction")
    axes[0].axhline(0.5, color="#b45309", linestyle="--", label="Target = 0.5")
    axes[0].set(ylabel="Network output")
    axes[1].plot(history[:, 0], history[:, 2], "o-", label="SSE")
    axes[1].axhline(TOLERANCE, color="#b45309", linestyle="--", label=f"Threshold = {TOLERANCE}")
    axes[1].set(ylabel="Sum of squared errors", ylim=(0, None))
    for ax in axes:
        ax.set(xlabel="Number of updates", xticks=history[:, 0])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(OUTPUT / "training.svg", metadata={"Date": None})
    plt.close(fig)


started = perf_counter()
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "bp-lab8"})
inputs = np.array([[0.35], [0.9]])
targets = np.array([[0.5]])
params = initialize()
print(f"BP 网络：2 -> 2 -> 1；随机种子 {SEED}；学习率 {LEARNING_RATE}")
check_gradients(inputs, targets, params)
history = train(inputs, targets, params)
np.savetxt(OUTPUT / "history.csv", history, delimiter=",", header="updates,prediction,squared_error", comments="")
np.savez(OUTPUT / "model.npz", allow_pickle=False, **params, inputs=inputs, targets=targets)
plot_history(history)
elapsed = perf_counter() - started
with (OUTPUT / "results.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["updates", "target", "prediction", "squared_error", "threshold", "seconds"])
    writer.writerow([int(history[-1, 0]), targets.item(), history[-1, 1], history[-1, 2], TOLERANCE, elapsed])
print(f"平方和误差 < {TOLERANCE}：通过；运行用时 {elapsed:.3f} 秒；输出目录：{OUTPUT}")
