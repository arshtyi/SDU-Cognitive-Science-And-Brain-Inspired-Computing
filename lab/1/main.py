import csv
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent
matplotlib.use("Agg")
type Array = NDArray[np.float64]
type Scores = list[tuple[int, int, int]]
SEED = 42
TRIALS = 100
FLIPS = (0, 1, 2, 3, 6, 9)
OUTPUT = ROOT / "output"
DIGITS = (
    "01110 10001 10001 10001 10001 01110",  # 0
    "00100 01100 00100 00100 00100 01110",  # 1
    "01110 10001 00010 00100 01000 11111",  # 2
    "11110 00001 00110 00001 10001 01110",  # 3
    "00010 00110 01010 10010 11111 00010",  # 4
    "11111 10000 11110 00001 00001 11110",  # 5
    "00110 01000 10000 11110 10001 01110",  # 6
    "11111 00001 00010 00100 01000 01000",  # 7
    "01110 10001 01110 10001 10001 01110",  # 8
    "01110 10001 01111 00001 00010 01100",  # 9
)


def make_patterns() -> Array:
    return np.array(
        [[1 if c == "1" else -1 for c in s.replace(" ", "")] for s in DIGITS],
        dtype=np.float64,
    )


def train(
    patterns: Array,
    rule: Literal["hebb", "projection"] = "projection",
) -> Array:
    if rule == "hebb":
        weights = patterns.T @ patterns / patterns.shape[1]
    else:
        weights = np.linalg.pinv(patterns) @ patterns
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0)
    return weights


def add_noise(pattern: Array, flips: int, rng: np.random.Generator) -> Array:
    noisy = pattern.copy()
    indices = rng.choice(pattern.size, flips, replace=False)
    noisy[indices] *= -1
    return noisy


def energy(weights: Array, state: Array) -> float:
    return float(-0.5 * state @ weights @ state)


def recall(weights: Array, initial: Array, max_sweeps: int = 100) -> tuple[Array, list[float]]:
    state = initial.copy()
    energies = [energy(weights, state)]
    for _ in range(max_sweeps):
        previous = state.copy()
        for i in range(state.size):
            field = float(weights[i] @ state)
            if abs(field) > 1e-10:
                state[i] = 1 if field > 0 else -1
            energies.append(energy(weights, state))
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
        correct = 0
        for pattern in patterns:
            for _ in range(repeats):
                noisy = add_noise(pattern, flips, rng)
                output, _ = recall(weights, noisy)
                correct += int(np.array_equal(output, pattern))
        scores.append((flips, repeats * len(patterns), correct))
    return scores


def plot_recall(patterns: Array, weights: Array) -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 10, figsize=(11, 4.2), layout="constrained")
    print("\n10% 噪声示例（? 表示未恢复为已存储数字）：")
    for digit, pattern in enumerate(patterns):
        noisy = add_noise(pattern, 3, rng)
        output, _ = recall(weights, noisy)
        label = recognize(patterns, output)
        name = "?" if label is None else str(label)
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


def plot_metrics(
    scores: dict[str, Scores],
    patterns: Array,
    weights: Array,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for rule, rows in scores.items():
        axes[0].plot(
            [100 * flips / 30 for flips, _, _ in rows],
            [100 * hits / total for _, total, hits in rows],
            "o-",
            label=rule,
        )
    axes[0].set(
        xlabel="Flipped pixels (%)",
        ylabel="Exact recovery (%)",
        ylim=(-3, 103),
    )
    axes[0].legend()
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


def save_scores(scores: dict[str, Scores]) -> None:
    path = OUTPUT / "results.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["flips", "total", "hebb", "projection"])
        rows = zip(
            scores["hebb"],
            scores["projection"],
            strict=True,
        )
        print("\n翻转数  样本数  Hebb 恢复率  伪逆恢复率")
        for (flips, total, hebb), (_, _, proj) in rows:
            writer.writerow([flips, total, hebb, proj])
            print(f"{flips:6d} {total:7d} {hebb / total:11.1%} {proj / total:11.1%}")


patterns = make_patterns()
weights = train(patterns)
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "hopfield-lab1"})
scores = {
    "hebb": evaluate(patterns, train(patterns, "hebb")),
    "projection": evaluate(patterns, weights),
}
print(f"Hopfield：30 个神经元，随机种子 {SEED}")
save_scores(scores)
plot_recall(patterns, weights)
plot_metrics(scores, patterns, weights)
