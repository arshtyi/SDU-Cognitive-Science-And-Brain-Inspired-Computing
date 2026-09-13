import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

type Array = NDArray[np.float64]
type Scores = list[tuple[int, int, int, int, int]]

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SEED = 42
LEARNING_RATE = 1 / 30
TRIALS = 100
FLIPS = (0, 1, 2, 3, 6, 9)
DIGITS = [
    "01110 10001 10001 10001 10001 01110",  # 0
    "00100 01100 00100 00100 00100 01110",  # 1
    "01110 10001 00010 00100 01000 11111",  # 2
]
PATTERNS = [[1 if c == "1" else -1 for c in digit.replace(" ", "")] for digit in DIGITS]


def train(patterns: Array) -> Array:
    rng = np.random.default_rng(SEED)
    size = patterns.shape[1]
    weights = rng.uniform(-0.001, 0.001, (size, size))
    weights = (weights + weights.T) / 2
    for pattern in patterns:
        weights += LEARNING_RATE * np.outer(pattern, pattern)
    np.fill_diagonal(weights, 0)
    return weights


def add_noise(pattern: Array, flips: int, rng: np.random.Generator) -> Array:
    noisy = pattern.copy()
    indices = rng.choice(pattern.size, flips, replace=False)
    noisy[indices] *= -1
    return noisy


def recall(weights: Array, initial: Array, max_sweeps: int = 100) -> tuple[Array, list[float]]:
    state = initial.copy()
    energies = [float(-0.5 * state @ weights @ state)]
    for _ in range(max_sweeps):
        previous = state.copy()
        for i in range(state.size):
            field = float(weights[i] @ state)
            if abs(field) > 1e-10:
                state[i] = 1 if field > 0 else -1
            energies.append(float(-0.5 * state @ weights @ state))
        if np.array_equal(state, previous):
            return state, energies
    raise RuntimeError("在最大更新轮数内未收敛")


def recognize(patterns: Array, state: Array) -> int | None:
    matches = np.flatnonzero(np.all(patterns == state, axis=1))
    return int(matches[0]) if matches.size else None


def evaluate(patterns: Array, weights: Array) -> Scores:
    rng = np.random.default_rng(SEED)
    scores = []
    for flips in FLIPS:
        repeats = 1 if flips == 0 else TRIALS
        correct = wrong = unknown = 0
        for digit, pattern in enumerate(patterns):
            for _ in range(repeats):
                noisy = add_noise(pattern, flips, rng)
                output, _ = recall(weights, noisy)
                label = recognize(patterns, output)
                correct += int(label == digit)
                wrong += int(label is not None and label != digit)
                unknown += int(label is None)
        total = len(patterns) * repeats
        scores.append((flips, total, correct, wrong, unknown))
    return scores


def plot_recall(patterns: Array, weights: Array) -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 3, figsize=(5, 5.4), layout="constrained")
    path = OUTPUT / "examples.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["digit", "noisy", "output", "prediction"])
        print("\n10% 噪声示例（? 表示未匹配到存储数字）：")
        for digit, pattern in enumerate(patterns):
            noisy = add_noise(pattern, 3, rng)
            output, _ = recall(weights, noisy)
            label = recognize(patterns, output)
            name = "?" if label is None else str(label)
            writer.writerow([digit, noisy.tolist(), output.tolist(), name])
            print(f"  {digit} -> {name}")
            for row, state in enumerate((pattern, noisy, output)):
                ax = axes[row, digit]
                ax.imshow(
                    state.reshape(6, 5),
                    cmap="binary",
                    vmin=-1,
                    vmax=1,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            axes[0, digit].set_title(str(digit))
            axes[2, digit].set_xlabel(f"-> {name}")
    for row, name in enumerate(("Original", "Noisy", "Recall")):
        axes[row, 0].set_ylabel(name)
    fig.savefig(OUTPUT / "recall.svg", metadata={"Date": None})
    plt.close(fig)


def plot_metrics(scores: Scores, patterns: Array, weights: Array) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    axes[0].plot(
        [100 * flips / 30 for flips, *_ in scores],
        [100 * hits / total for _, total, hits, _, _ in scores],
        "o-",
    )
    axes[0].set(
        xlabel="Flipped pixels (%)",
        ylabel="Exact recovery (%)",
        ylim=(-3, 103),
    )
    rng = np.random.default_rng(SEED)
    noisy = add_noise(patterns[0], 3, rng)
    _, energies = recall(weights, noisy)
    axes[1].plot(energies, color="#b45309")
    axes[1].set(
        xlabel="Neuron updates",
        ylabel="Energy",
        title="Digit 0, three flipped pixels",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(OUTPUT / "metrics.svg", metadata={"Date": None})
    plt.close(fig)


patterns = np.array(PATTERNS, dtype=np.float64)
weights = train(patterns)
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "hebb-lab3"})
np.savetxt(OUTPUT / "weights.csv", weights, delimiter=",")
scores = evaluate(patterns, weights)
print(f"Hebb：30 个神经元，数字 0–2，随机种子 {SEED}")
print("翻转数  样本数  完整恢复  错误数字  未匹配  恢复率")
path = OUTPUT / "results.csv"
with path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["flips", "total", "correct", "wrong", "unknown"])
    writer.writerows(scores)
    for flips, total, correct, wrong, unknown in scores:
        print(f"{flips:6d} {total:7d} {correct:9d} {wrong:9d} {unknown:7d} {correct / total:8.1%}")
plot_recall(patterns, weights)
plot_metrics(scores, patterns, weights)
