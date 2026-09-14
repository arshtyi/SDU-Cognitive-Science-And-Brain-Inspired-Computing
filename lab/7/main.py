import csv
import hashlib
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

matplotlib.use("Agg")

type Integers = NDArray[np.int64]
type History = list[tuple[int, float, float]]

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def load_data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
            torch.from_numpy(data["x_train"]).unsqueeze(1).float() / 255,
            torch.from_numpy(data["y_train"].astype(np.int64)),
            torch.from_numpy(data["x_test"]).unsqueeze(1).float() / 255,
            torch.from_numpy(data["y_test"].astype(np.int64)),
        )


def build_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def train(model: nn.Sequential, images: Tensor, labels: Tensor) -> History:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED)
    history = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(images), generator=generator)
        total_loss, correct = 0.0, 0
        for indices in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images[indices])
            loss = criterion(logits, labels[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
            correct += int((logits.detach().argmax(dim=1) == labels[indices]).sum().item())
        history.append((epoch, total_loss / len(images), correct / len(images)))
        print(f"Epoch {epoch}: online train CE={history[-1][1]:.6f}, accuracy={history[-1][2]:.2%}", flush=True)
    return history


@torch.inference_mode()
def evaluate(model: nn.Sequential, images: Tensor, labels: Tensor) -> tuple[float, Integers]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    predictions = np.empty(len(labels), dtype=np.int64)
    total_loss = 0.0
    for start in range(0, len(images), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        logits = model(images[batch])
        total_loss += criterion(logits, labels[batch]).item()
        predictions[batch] = logits.argmax(dim=1).numpy()
    return total_loss / len(labels), predictions


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


def plot_training(history: History) -> None:
    values = np.array(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for ax, column, label in zip(axes, (1, 2), ("Online training cross-entropy", "Online training accuracy"), strict=True):
        ax.plot(values[:, 0], values[:, column], "o-")
        ax.set(xlabel="Epoch", ylabel=label, xticks=values[:, 0])
        ax.grid(alpha=0.2)
    fig.savefig(OUTPUT / "training.svg", metadata={"Date": None})
    plt.close(fig)


def plot_confusion(labels: Integers, predictions: Integers) -> None:
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


def plot_examples(images: Tensor, labels: Integers, predictions: Integers) -> None:
    fig, axes = plt.subplots(2, 10, figsize=(11, 3), layout="constrained")
    examples = np.concatenate((np.arange(10), np.flatnonzero(labels != predictions)[:10]))
    for ax, index in zip(axes.flat, examples, strict=False):
        ax.imshow(images[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"#{index}: {labels[index]} -> {predictions[index]}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(OUTPUT / "examples.svg", metadata={"Date": None})
    plt.close(fig)


started = perf_counter()
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.hashsalt": "cnn-lab7"})
torch.manual_seed(SEED)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
train_images, train_labels, test_images, test_labels = load_data()
mean = train_images.mean().item()
std = train_images.std(correction=0).item()
train_images = (train_images - mean) / std
test_inputs = (test_images - mean) / std
model = build_model()
print(f"CNN: {sum(p.numel() for p in model.parameters())} 个参数；CPU；随机种子 {SEED}", flush=True)
print(f"训练 {len(train_images)} 张，测试 {len(test_images)} 张；mean={mean:.6f}, std={std:.6f}", flush=True)
prepared = perf_counter()
history = train(model, train_images, train_labels)
trained = perf_counter()
train_loss, train_predictions = evaluate(model, train_images, train_labels)
test_loss, test_predictions = evaluate(model, test_inputs, test_labels)
evaluated = perf_counter()
torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "seed": SEED, "epochs": EPOCHS}, OUTPUT / "model.pt")
np.savetxt(OUTPUT / "history.csv", history, delimiter=",", header="epoch,online_train_loss,online_train_accuracy", comments="")
save_results(train_labels.numpy(), train_predictions, test_labels.numpy(), test_predictions)
with (OUTPUT / "loss.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["split", "cross_entropy"])
    writer.writerows((("train", train_loss), ("test", test_loss)))
plot_training(history)
plot_confusion(test_labels.numpy(), test_predictions)
plot_examples(test_images, test_labels.numpy(), test_predictions)
elapsed = perf_counter() - started
with (OUTPUT / "timing.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["stage", "seconds"])
    writer.writerows(
        (
            ("prepare", prepared - started),
            ("training", trained - prepared),
            ("evaluation", evaluated - trained),
            ("output", elapsed - (evaluated - started)),
            ("total", elapsed),
        )
    )
test_accuracy = float(np.mean(test_predictions == test_labels.numpy()))
print(f"测试准确率：{test_accuracy:.2%}；手册要求 ≥90%：{'通过' if test_accuracy >= 0.9 else '未通过'}")
print(f"运行用时：{elapsed:.2f} 秒；输出目录：{OUTPUT}")
if test_accuracy < 0.9:
    raise RuntimeError("测试准确率未达到手册要求的 90%")
