import csv
import hashlib
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

type Array = NDArray[np.float64]
type Integers = NDArray[np.int64]

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SEED = 42
GRID_SIZE = 10
COMPONENTS = GRID_SIZE**2
EPOCHS = 200
LEARNING_RATE = 0.5
EPSILON = 1e-8
RIDGE = 1.0
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def load_data() -> tuple[Array, Integers, Array, Integers]:
    path = OUTPUT / "mnist.npz"
    if not path.exists():
        print("下载 MNIST……", flush=True)
        temporary = path.with_suffix(".part")
        urlretrieve(MNIST_URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != MNIST_SHA256:
            raise ValueError("MNIST 下载校验失败")
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MNIST_SHA256:
        raise ValueError("MNIST 文件校验失败，请删除 output/mnist.npz 后重新运行")
    with np.load(path, allow_pickle=False) as data:
        return (
            data["x_train"].reshape(-1, 784).astype(np.float64) / 255,
            data["y_train"].astype(np.int64),
            data["x_test"].reshape(-1, 784).astype(np.float64) / 255,
            data["y_test"].astype(np.int64),
        )


def whiten(images: Array) -> tuple[Array, Array, Array, Array, float]:
    mean = images.mean(axis=0)
    centered = images - mean
    values, vectors = np.linalg.eigh(centered.T @ centered / len(images))
    indices = np.argsort(values)[-COMPONENTS:][::-1]
    selected = values[indices]
    if selected.min() <= EPSILON:
        raise ValueError("训练数据的有效秩不足以支持指定的成分数")
    whitening = vectors[:, indices] / np.sqrt(selected)
    dewhitening = (vectors[:, indices] * np.sqrt(selected)).T
    retained = float(selected.sum() / values.sum())
    return centered @ whitening, mean, whitening, dewhitening, retained


def make_neighborhood() -> Array:
    coordinates = np.array(list(np.ndindex(GRID_SIZE, GRID_SIZE)))
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    distance = np.minimum(distance, GRID_SIZE - distance)
    neighbors = np.all(distance <= 1, axis=2).astype(np.float64)
    return neighbors / neighbors.sum(axis=1, keepdims=True)


def orthogonalize(weights: Array) -> Array:
    left, _, right = np.linalg.svd(weights, full_matrices=False)
    return left @ right


def loss_gradient(data: Array, weights: Array, neighborhood: Array) -> tuple[float, Array]:
    sources = data @ weights.T
    amplitudes = np.sqrt(sources**2 @ neighborhood.T + EPSILON)
    loss = float(amplitudes.sum(axis=1).mean())
    gradient = (sources * ((1 / amplitudes) @ neighborhood)).T @ data / len(data)
    return loss, gradient


def train(data: Array, neighborhood: Array) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(SEED)
    weights = orthogonalize(rng.normal(size=(COMPONENTS, COMPONENTS)))
    initial = weights.copy()
    history = []
    for epoch in range(EPOCHS + 1):
        loss, gradient = loss_gradient(data, weights, neighborhood)
        history.append((epoch, loss))
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: TopoICA objective={loss:.6f}", flush=True)
        if epoch < EPOCHS:
            weights = orthogonalize(weights - LEARNING_RATE * gradient)
    return weights, initial, np.array(history)


def extract(data: Array, weights: Array, neighborhood: Array) -> Array:
    sources = data @ weights.T
    amplitudes = np.sqrt(sources**2 @ neighborhood.T + EPSILON)
    return np.column_stack((sources, amplitudes))


def fit_classifier(features: Array, labels: Integers) -> tuple[Array, Array, Array]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), EPSILON)
    design = np.column_stack(((features - mean) / scale, np.ones(len(features))))
    penalty = RIDGE * np.eye(design.shape[1])
    penalty[-1, -1] = 0  # 截距不参与正则化。
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ np.eye(10)[labels])
    return mean, scale, weights


def predict(features: Array, mean: Array, scale: Array, weights: Array) -> Integers:
    scores = ((features - mean) / scale) @ weights[:-1] + weights[-1]
    return scores.argmax(axis=1).astype(np.int64)


def save_results(train_labels: Integers, train_predictions: Integers, test_labels: Integers, test_predictions: Integers) -> None:
    with (OUTPUT / "results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "digit", "total", "correct", "accuracy"])
        for name, labels, predictions in (("train", train_labels, train_predictions), ("test", test_labels, test_predictions)):
            for digit in ("all", *range(10)):
                mask = np.ones(len(labels), dtype=bool) if digit == "all" else labels == digit
                total = int(mask.sum())
                correct = int(np.count_nonzero(labels[mask] == predictions[mask]))
                writer.writerow([name, digit, total, correct, correct / total])
                print(f"{name:5s} {str(digit):3s}: {correct}/{total} = {correct / total:.2%}")
    with (OUTPUT / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "target", "prediction"])
        writer.writerows(zip(range(len(test_labels)), test_labels, test_predictions, strict=True))


def save_diagnostics(data: Array, initial: Array, weights: Array, neighborhood: Array, split: str) -> list[tuple[str, str, float, float, float]]:
    pairs = np.triu(np.ones(neighborhood.shape, dtype=bool), k=1)
    near = pairs & (neighborhood > 0)
    far = pairs & (neighborhood == 0)
    rows = []
    for stage, matrix in (("initial", initial), ("trained", weights)):
        sources = data @ matrix.T
        correlation = np.corrcoef(sources**2, rowvar=False)
        near_mean = float(correlation[near].mean())
        far_mean = float(correlation[far].mean())
        loss = float(np.sqrt(sources**2 @ neighborhood.T + EPSILON).sum(axis=1).mean())
        rows.append((split, stage, loss, near_mean, far_mean))
        np.savetxt(OUTPUT / f"energy_correlation_{split}_{stage}.csv", correlation, delimiter=",")
    return rows


def plot_model(whitening: Array, weights: Array, history: Array) -> None:
    filters = (weights @ whitening.T).reshape(GRID_SIZE, GRID_SIZE, 28, 28)
    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(9, 9))
    for ax, kernel in zip(axes.flat, filters.reshape(-1, 28, 28), strict=True):
        limit = np.abs(kernel).max()
        ax.imshow(kernel, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(OUTPUT / "filters.svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    ax.plot(history[:, 0], history[:, 1])
    ax.set(xlabel="Full-batch update", ylabel="Mean sum of neighborhood amplitudes")
    ax.grid(alpha=0.2)
    fig.savefig(OUTPUT / "training.svg", metadata={"Date": None})
    plt.close(fig)


def plot_predictions(images: Array, labels: Integers, predictions: Integers) -> None:
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    np.savetxt(OUTPUT / "confusion.csv", confusion, delimiter=",", fmt="%d")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.imshow(confusion, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=8, color="white" if confusion[i, j] > 500 else "black")
    ax.set(xlabel="Prediction", ylabel="True digit", xticks=range(10), yticks=range(10))
    fig.savefig(OUTPUT / "confusion.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(2, 10, figsize=(11, 3), layout="constrained")
    examples = np.concatenate((np.arange(10), np.flatnonzero(labels != predictions)[:10]))
    for ax, index in zip(axes.flat, examples, strict=False):
        ax.imshow(images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"#{index}: {labels[index]} -> {predictions[index]}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(OUTPUT / "examples.svg", metadata={"Date": None})
    plt.close(fig)


started = perf_counter()
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "topoica-lab6"})
train_images, train_labels, test_images, test_labels = load_data()
train_data, pixel_mean, whitening, dewhitening, retained = whiten(train_images)
neighborhood = make_neighborhood()
print(f"TopoICA: {GRID_SIZE}×{GRID_SIZE} 网格，3×3 环面邻域，随机种子 {SEED}", flush=True)
print(f"PCA 保留方差：{retained:.4%}；训练 {len(train_data)} 张，测试 {len(test_images)} 张", flush=True)
weights, initial, history = train(train_data, neighborhood)
train_features = extract(train_data, weights, neighborhood)
feature_mean, feature_scale, classifier = fit_classifier(train_features, train_labels)
trained = perf_counter()
test_data = (test_images - pixel_mean) @ whitening
test_features = extract(test_data, weights, neighborhood)
train_predictions = predict(train_features, feature_mean, feature_scale, classifier)
test_predictions = predict(test_features, feature_mean, feature_scale, classifier)
np.savez(
    OUTPUT / "model.npz",
    pixel_mean=pixel_mean,
    whitening=whitening,
    dewhitening=dewhitening,
    weights=weights,
    initial=initial,
    neighborhood=neighborhood,
    feature_mean=feature_mean,
    feature_scale=feature_scale,
    classifier=classifier,
    retained_variance=retained,
)
np.savetxt(OUTPUT / "history.csv", history, delimiter=",", header="epoch,objective", comments="")
save_results(train_labels, train_predictions, test_labels, test_predictions)
diagnostics = save_diagnostics(train_data, initial, weights, neighborhood, "train")
diagnostics += save_diagnostics(test_data, initial, weights, neighborhood, "test")
with (OUTPUT / "topology.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["split", "stage", "objective", "near_energy_correlation", "far_energy_correlation"])
    writer.writerows(diagnostics)
plot_model(whitening, weights, history)
plot_predictions(test_images, test_labels, test_predictions)
elapsed = perf_counter() - started
with (OUTPUT / "timing.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["stage", "seconds"])
    writer.writerows((("training", trained - started), ("test_and_output", elapsed - (trained - started)), ("total", elapsed)))
print(f"运行用时：{elapsed:.2f} 秒；输出目录：{OUTPUT}")
