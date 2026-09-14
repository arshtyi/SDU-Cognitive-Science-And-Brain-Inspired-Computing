import csv
from pathlib import Path
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

type Array = NDArray[np.float64]
type Integers = NDArray[np.int64]
type Parameters = dict[str, Array]
type Cache = list[tuple[Array, Array, Array, Array, Array, Array, Array, Array]]

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SEED = 42
BITS = 8
LIMIT = 128
TRAIN_SIZE = 12_288
VALID_SIZE = 2_048
HIDDEN_SIZE = 16
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.01


def encode(values: Integers) -> Array:
    return ((values[..., None] >> np.arange(BITS)) & 1).astype(np.float64)


def make_data() -> tuple[Integers, Array, Array]:
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(LIMIT**2)
    pairs = np.column_stack((indices // LIMIT, indices % LIMIT))
    inputs = encode(pairs).transpose(2, 0, 1)
    targets = encode(pairs.sum(axis=1)).T[..., None]
    return pairs, inputs, targets


def initialize() -> Parameters:
    rng = np.random.default_rng(SEED)
    width = HIDDEN_SIZE + 2
    params = {
        "W": rng.normal(0, 1 / np.sqrt(width), (width, 4 * HIDDEN_SIZE)),
        "b": np.zeros(4 * HIDDEN_SIZE),
        "Wy": rng.normal(0, 1 / np.sqrt(HIDDEN_SIZE), (HIDDEN_SIZE, 1)),
        "by": np.zeros(1),
    }
    params["b"][:HIDDEN_SIZE] = 1  # 遗忘门初始偏置。
    return params


def sigmoid(values: Array) -> Array:
    return 0.5 * (1 + np.tanh(values / 2))


def forward(inputs: Array, params: Parameters) -> tuple[Array, Cache]:
    h = np.zeros((inputs.shape[1], params["Wy"].shape[0]))
    c = np.zeros_like(h)
    logits = np.empty((*inputs.shape[:2], 1))
    cache = []
    for t, x in enumerate(inputs):
        z = np.concatenate((x, h), axis=1)
        af, ai, ag, ao = np.split(z @ params["W"] + params["b"], 4, axis=1)
        f, i, g, o = sigmoid(af), sigmoid(ai), np.tanh(ag), sigmoid(ao)
        previous_c = c
        c = f * c + i * g
        h = o * np.tanh(c)
        logits[t] = h @ params["Wy"] + params["by"]
        cache.append((z, f, i, g, o, previous_c, c, h))
    return logits, cache


def backward(logits: Array, targets: Array, cache: Cache, params: Parameters) -> Parameters:
    gradients = {name: np.zeros_like(value) for name, value in params.items()}
    dh = np.zeros_like(cache[0][-1])
    dc = np.zeros_like(dh)
    errors = (sigmoid(logits) - targets) / targets.size
    for t in reversed(range(len(cache))):
        z, f, i, g, o, previous_c, c, h = cache[t]
        gradients["Wy"] += h.T @ errors[t]
        gradients["by"] += errors[t].sum(axis=0)
        dh = dh + errors[t] @ params["Wy"].T
        tanh_c = np.tanh(c)
        dc = dc + dh * o * (1 - tanh_c**2)
        gates = np.concatenate(
            (
                dc * previous_c * f * (1 - f),
                dc * g * i * (1 - i),
                dc * i * (1 - g**2),
                dh * tanh_c * o * (1 - o),
            ),
            axis=1,
        )
        gradients["W"] += z.T @ gates
        gradients["b"] += gates.sum(axis=0)
        dh = (gates @ params["W"].T)[:, 2:]
        dc = dc * f
    return gradients


def evaluate(inputs: Array, targets: Array, params: Parameters) -> tuple[float, float, float, Integers]:
    predictions = np.empty(targets.shape, dtype=np.int64)
    loss = 0.0
    for start in range(0, inputs.shape[1], BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        logits, _ = forward(inputs[:, batch], params)
        loss += float(np.sum(np.logaddexp(0, logits) - targets[:, batch] * logits))
        predictions[:, batch] = logits >= 0  # 等价于 sigmoid 输出 >= 0.5。
    correct = predictions == targets
    sums = predictions[:, :, 0].T @ (1 << np.arange(BITS))
    return loss / targets.size, float(correct.mean()), float(correct.all(axis=0).mean()), sums


def train(inputs: Array, targets: Array, params: Parameters) -> list[tuple[int, float, float, float]]:
    rng = np.random.default_rng(SEED)
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    valid = slice(TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE)
    history = []
    step = 0
    for epoch in range(EPOCHS + 1):
        if epoch:
            order = rng.permutation(TRAIN_SIZE)
            for start in range(0, TRAIN_SIZE, BATCH_SIZE):
                batch = order[start : start + BATCH_SIZE]
                logits, cache = forward(inputs[:, batch], params)
                gradients = backward(logits, targets[:, batch], cache, params)
                norm = np.sqrt(sum(float(np.sum(g**2)) for g in gradients.values()))
                step += 1
                for name, value in params.items():
                    gradient = gradients[name] / max(1.0, norm)
                    first[name] = 0.9 * first[name] + 0.1 * gradient
                    second[name] = 0.999 * second[name] + 0.001 * gradient**2
                    m = first[name] / (1 - 0.9**step)
                    v = second[name] / (1 - 0.999**step)
                    value -= LEARNING_RATE * m / (np.sqrt(v) + 1e-8)
        train_loss, _, _, _ = evaluate(inputs[:, :TRAIN_SIZE], targets[:, :TRAIN_SIZE], params)
        valid_loss, _, valid_exact, _ = evaluate(inputs[:, valid], targets[:, valid], params)
        history.append((epoch, train_loss, valid_loss, valid_exact))
        print(f"Epoch {epoch:2d}: train BCE={train_loss:.6f}, valid BCE={valid_loss:.6f}, valid exact={valid_exact:.2%}")
    return history


def save_results(pairs: Integers, inputs: Array, targets: Array, params: Parameters) -> None:
    splits = np.array(["train"] * TRAIN_SIZE + ["validation"] * VALID_SIZE + ["test"] * (len(pairs) - TRAIN_SIZE - VALID_SIZE))
    _, _, _, predictions = evaluate(inputs, targets, params)
    expected = pairs.sum(axis=1)
    with (OUTPUT / "results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "total", "correct", "bit_accuracy", "exact_accuracy", "bce"])
        for name in ("train", "validation", "test", "all"):
            mask = np.ones(len(pairs), dtype=bool) if name == "all" else splits == name
            subset_loss, bit, exact, _ = evaluate(inputs[:, mask], targets[:, mask], params)
            total = int(mask.sum())
            correct = int(np.count_nonzero(predictions[mask] == expected[mask]))
            writer.writerow([name, total, correct, bit, exact, subset_loss])
            print(f"{name:10s}: {correct}/{total}, bit={bit:.2%}, exact={exact:.2%}, BCE={subset_loss:.6f}")
    with (OUTPUT / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "a", "b", "target", "prediction", "a_binary", "b_binary", "target_binary", "prediction_binary"])
        for (a, b), target, prediction, split in zip(pairs, expected, predictions, splits, strict=True):
            writer.writerow([split, a, b, target, prediction, f"{a:08b}", f"{b:08b}", f"{target:08b}", f"{prediction:08b}"])
    examples = ((0, 0), (0, 127), (1, 1), (5, 10), (15, 1), (63, 1), (85, 43), (127, 1), (127, 127))
    with (OUTPUT / "examples.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["a", "b", "target", "prediction", "prediction_binary", "split"])
        for a, b in examples:
            index = int(np.flatnonzero(np.all(pairs == (a, b), axis=1))[0])
            prediction = predictions[index]
            writer.writerow([a, b, a + b, prediction, f"{prediction:08b}", splits[index]])
            print(f"  {a:3d} + {b:3d} = {prediction:3d} ({prediction:08b}), expected {a + b}")


def plot_history(history: list[tuple[int, float, float, float]]) -> None:
    rows = np.array(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    axes[0].semilogy(rows[:, 0], rows[:, 1], label="Train")
    axes[0].semilogy(rows[:, 0], rows[:, 2], "--", label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[0].legend()
    axes[1].plot(rows[:, 0], 100 * rows[:, 3], color="#b45309")
    axes[1].set(xlabel="Epoch", ylabel="Validation exact sum (%)", ylim=(-3, 103))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(OUTPUT / "training.svg", metadata={"Date": None})
    plt.close(fig)


started = perf_counter()
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "lstm-lab4"})
pairs, inputs, targets = make_data()
params = initialize()
print(f"LSTM：2 -> {HIDDEN_SIZE} -> 1，八位二进制加法，随机种子 {SEED}")
history = train(inputs, targets, params)
with (OUTPUT / "history.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "train_bce", "validation_bce", "validation_exact"])
    writer.writerows(history)
np.savez(OUTPUT / "model.npz", allow_pickle=False, **params)
save_results(pairs, inputs, targets, params)
plot_history(history)
print(f"运行用时：{perf_counter() - started:.2f} 秒；输出目录：{OUTPUT}")
