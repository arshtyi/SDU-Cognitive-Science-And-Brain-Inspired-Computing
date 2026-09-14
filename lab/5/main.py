import csv
import hashlib
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

matplotlib.use("Agg")

type Array = NDArray[np.float64]
type Integers = NDArray[np.int64]

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SEED = 42
SIZES = tuple(range(7, 38, 2))
DIRECTIONS = (0, 45, 90, 135)
PATCH_SIZE = 3
PATCHES_PER_CLASS = 20
BATCH_SIZE = 32
RBF_SIGMA = 1.0
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
            data["x_train"].astype(np.float64) / 255,
            data["y_train"].astype(np.int64),
            data["x_test"].astype(np.float64) / 255,
            data["y_test"].astype(np.int64),
        )


def make_filters() -> Array:
    filters = np.zeros((16, 4, 37, 37))
    for scale, size in enumerate(SIZES):
        radius = size // 2
        y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        sigma = 0.3 * size
        for orientation, degrees in enumerate(DIRECTIONS):
            theta = np.deg2rad(degrees)
            u = x * np.cos(theta) + y * np.sin(theta)
            v = -x * np.sin(theta) + y * np.cos(theta)
            kernel = np.exp(-(u**2 + 0.5**2 * v**2) / (2 * sigma**2)) * np.cos(2 * np.pi * u / (2.5 * sigma))
            kernel -= kernel.mean()
            kernel /= np.linalg.norm(kernel)
            filters[scale, orientation, 18 - radius : 19 + radius, 18 - radius : 19 + radius] = kernel
    return filters


def s1(images: Array, filters_fft: NDArray[np.complex128]) -> Array:
    image_fft = np.fft.rfft2(images, s=(64, 64), axes=(-2, -1))
    integral = np.pad(images**2, ((0, 0), (1, 0), (1, 0))).cumsum(axis=1).cumsum(axis=2)
    responses = np.empty((len(images), 16, 4, 28, 28))
    for scale, size in enumerate(SIZES):
        low = np.clip(np.arange(28) - size // 2, 0, 28)
        high = np.clip(np.arange(28) + size // 2 + 1, 0, 28)
        energy = integral[:, high[:, None], high] - integral[:, low[:, None], high] - integral[:, high[:, None], low] + integral[:, low[:, None], low]
        convolution = np.fft.irfft2(image_fft[:, None] * filters_fft[scale], s=(64, 64), axes=(-2, -1))
        responses[:, scale] = np.abs(convolution[:, :, 18:46, 18:46]) / np.maximum(np.sqrt(np.maximum(energy, 0)), 1e-8)[:, None]
    return np.clip(responses, 0, 1)


def c1(responses: Array) -> Array:
    windows = sliding_window_view(responses, (8, 8), axis=(-2, -1))
    pooled = windows[..., ::4, ::4, :, :].max(axis=(-2, -1))
    return pooled.reshape(len(responses), 8, 2, 4, 6, 6).max(axis=2)


def sample_patches(images: Array, labels: Integers, filters_fft: NDArray[np.complex128]) -> tuple[Array, Integers]:
    rng = np.random.default_rng(SEED)
    patches, sources = [], []
    for digit in range(10):
        indices = rng.choice(np.flatnonzero(labels == digit), PATCHES_PER_CLASS, replace=False)
        maps = c1(s1(images[indices], filters_fft))
        for index, maps_i in zip(indices, maps, strict=True):
            band = int(rng.integers(8))
            y, x = rng.integers(7 - PATCH_SIZE, size=2)
            patches.append(maps_i[band, :, y : y + PATCH_SIZE, x : x + PATCH_SIZE])
            sources.append((index, digit, band, y, x))
    return np.array(patches), np.array(sources, dtype=np.int64)


def s2(maps: Array, patches: Array) -> Array:
    windows = sliding_window_view(maps, (PATCH_SIZE, PATCH_SIZE), axis=(-2, -1))
    windows = windows.transpose(0, 1, 3, 4, 2, 5, 6).reshape(len(maps), 8, 4, 4, -1)
    templates = patches.reshape(len(patches), -1)
    distances = (windows**2).sum(axis=-1, keepdims=True) + (templates**2).sum(axis=1) - 2 * (windows @ templates.T)
    return np.exp(-np.maximum(distances, 0) / (2 * RBF_SIGMA**2))


def c2(responses: Array) -> Array:
    return responses.max(axis=(1, 2, 3))


def extract(images: Array, filters_fft: NDArray[np.complex128], patches: Array, name: str) -> Array:
    features = np.empty((len(images), len(patches)))
    for start in range(0, len(images), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        features[batch] = c2(s2(c1(s1(images[batch], filters_fft)), patches))
        if start % 2_000 == 0 or start + BATCH_SIZE >= len(images):
            print(f"{name} C2: {min(start + BATCH_SIZE, len(images))}/{len(images)}", flush=True)
    return features


def train(features: Array, labels: Integers) -> tuple[Array, Array, Array]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
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


def plot_results(images: Array, labels: Integers, predictions: Integers, filters: Array, patches: Array) -> None:
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
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"#{index}: {labels[index]} -> {predictions[index]}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(OUTPUT / "examples.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(6, 5), layout="constrained")
    for row, scale in enumerate((0, 5, 10, 15)):
        for col, degrees in enumerate(DIRECTIONS):
            ax = axes[row, col]
            limit = np.abs(filters[scale, col]).max()
            ax.imshow(filters[scale, col], cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_title(f"{SIZES[scale]} px, {degrees} deg", fontsize=8)
            ax.axis("off")
    fig.savefig(OUTPUT / "filters.svg", metadata={"Date": None})
    plt.close(fig)
    filters_fft = np.fft.rfft2(filters, s=(64, 64), axes=(-2, -1))
    first = s1(images[:1], filters_fft)
    pooled = c1(first)
    matched = s2(pooled, patches)
    feature = c2(matched)
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), layout="constrained")
    for ax, values, title in zip(
        axes[:4],
        (images[0], first[0, 0, 0], pooled[0, 0, 0], matched[0, 0, :, :, 0]),
        ("Input", "S1: scale 0, dir 0", "C1: band 0, dir 0", "S2: band 0, patch 0"),
        strict=True,
    ):
        ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    axes[4].plot(feature[0], linewidth=0.8)
    axes[4].set(title="C2: 200 features", xlabel="Patch", ylim=(0, 1))
    fig.savefig(OUTPUT / "layers.svg", metadata={"Date": None})
    plt.close(fig)


started = perf_counter()
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "hmax-lab5"})
train_images, train_labels, test_images, test_labels = load_data()
filters = make_filters()
filters_fft = np.fft.rfft2(filters, s=(64, 64), axes=(-2, -1))
patches, sources = sample_patches(train_images, train_labels, filters_fft)
np.savetxt(OUTPUT / "patch_sources.csv", sources, delimiter=",", fmt="%d", header="train_index,digit,band,y,x", comments="")
print(f"HMAX: 16 尺度 × 4 方向，8 子带，{len(patches)} 个训练模板，随机种子 {SEED}", flush=True)
train_features = extract(train_images, filters_fft, patches, "train")
mean, scale, weights = train(train_features, train_labels)
trained = perf_counter()
np.savez(OUTPUT / "model.npz", filters=filters, patches=patches, mean=mean, scale=scale, weights=weights)
test_features = extract(test_images, filters_fft, patches, "test")
train_predictions = predict(train_features, mean, scale, weights)
test_predictions = predict(test_features, mean, scale, weights)
np.savez(OUTPUT / "features.npz", train=train_features, test=test_features)
save_results(train_labels, train_predictions, test_labels, test_predictions)
plot_results(test_images, test_labels, test_predictions, filters, patches)
elapsed = perf_counter() - started
with (OUTPUT / "timing.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["stage", "seconds"])
    writer.writerows((("training", trained - started), ("test_and_output", elapsed - (trained - started)), ("total", elapsed)))
print(f"运行用时：{elapsed:.2f} 秒；输出目录：{OUTPUT}")
