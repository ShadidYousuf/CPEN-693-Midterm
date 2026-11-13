import sys
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchinfo import summary

import data

sys.stdout.reconfigure(encoding="utf-8")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent / "outputs"
LOGS, FIGURES, MODELS = OUT / "logs", OUT / "figures", OUT / "models"
for folder in (LOGS, FIGURES, MODELS):
    folder.mkdir(parents=True, exist_ok=True)


def get_log(path):
    file = open(path, "w", encoding="utf-8")

    def log(text=""):
        print(text, flush=True)
        file.write(text + "\n")
        file.flush()

    return log


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_step(model, loader, loss_fn, optimizer):
    model.train()
    total = 0
    for images, labels in loader:
        images, labels = data.prepare(images.to(DEVICE), loader.dataset.augment), labels.to(DEVICE)
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


def test_step(model, loader, loss_fn):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = data.prepare(images.to(DEVICE)), labels.to(DEVICE)
            outputs = model(images)
            total += loss_fn(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total / len(loader.dataset), 100 * correct / len(loader.dataset)


def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, log, patience=10):
    results = {"train_loss": [], "val_loss": [], "time": 0}
    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_step(model, train_loader, loss_fn, optimizer)
        scheduler.step()
        results["time"] += time.time() - start
        val_loss, val_acc = test_step(model, val_loader, loss_fn)
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        log(f"  epoch {epoch + 1:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.2f}")
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait == patience:
                log(f"  early stopping at epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)
    log(f"  restored weights of best epoch {results['val_loss'].index(best_loss) + 1}")
    return results


def get_summary(model, dataset):
    info = data.INFO[dataset]
    return str(summary(model, input_size=(1, info["channels"], info["size"], info["size"]), verbose=0))


def get_title(name, config):
    details = []
    if name == "fcn":
        details.append(f"{config['layers']} hidden layers")
    if name == "cnn":
        details.append(f"{config['layers']} conv layers")
    if "activation" in config:
        details.append(config["activation"])
    if config.get("batch_norm"):
        details.append("batch norm")
    if "dropout" in config:
        details.append(f"dropout {config['dropout']}")
    if config.get("pretrained"):
        details.append("ImageNet pretrained")
    if "weight_decay" in config:
        details.append(f"weight decay {config['weight_decay']}")
    if "label_smoothing" in config:
        details.append(f"label smoothing {config['label_smoothing']}")
    title = {"fcn": "FCN", "cnn": "CNN", "resnet18": "ResNet-18", "resnet50": "ResNet-50"}[name]
    return f"{title} [{', '.join(details)}]" if details else title


def plot_loss_curves(results, title, path):
    epochs = range(1, len(results["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results["train_loss"], label="training loss")
    plt.plot(epochs, results["val_loss"], label="validation loss")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
