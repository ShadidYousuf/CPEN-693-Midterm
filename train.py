import argparse

import torch
from torch import nn

import data
import models
import utils

CNN_BASE = {"layers": 5, "filters": 32, "stride": 1, "hidden": 256, "activation": "relu", "batch_norm": False, "dropout": 0.0}
RESNET = {}
PRETRAINED_RESNET = {"pretrained": True, "weight_decay": 0.05, "label_smoothing": 0.2}
CNN_CHANGES = [
    {},
    {"layers": 3},
    {"layers": 7},
    {"layers": 9},
    {"filters": 16},
    {"filters": 64},
    {"stride": 2},
    {"hidden": 128},
    {"hidden": 512},
    {"activation": "tanh"},
    {"activation": "leaky_relu"},
    {"activation": "gelu"},
    {"dropout": 0.1},
    {"dropout": 0.2},
    {"layers": 7, "filters": 64, "hidden": 512, "dropout": 0.2},
    {"layers": 5, "filters": 64, "hidden": 512, "activation": "gelu", "dropout": 0.2},
    {"layers": 9, "filters": 32, "hidden": 512, "activation": "leaky_relu", "dropout": 0.1},
    {"batch_norm": True},
    {"layers": 7, "batch_norm": True},
    {"layers": 9, "batch_norm": True},
    {"layers": 7, "filters": 64, "hidden": 512, "dropout": 0.2, "batch_norm": True},
]
FCN_FLOWER_CHANGES = [
    {},
    {"dropout": 0.1},
    {"dropout": 0.2},
    {"weight_decay": 0.05},
    {"label_smoothing": 0.1},
    {"label_smoothing": 0.2},
    {"weight_decay": 0.05, "label_smoothing": 0.1},
    {"weight_decay": 0.05, "label_smoothing": 0.2},
    {"dropout": 0.2, "weight_decay": 0.05, "label_smoothing": 0.1},
]
CNN_FLOWER_CHANGES = [
    {},
    {"dropout": 0.2},
    {"dropout": 0.5},
    {"weight_decay": 0.05},
    {"label_smoothing": 0.1},
    {"label_smoothing": 0.2},
    {"weight_decay": 0.05, "label_smoothing": 0.1},
    {"weight_decay": 0.05, "label_smoothing": 0.2},
    {"dropout": 0.5, "weight_decay": 0.05, "label_smoothing": 0.1},
]
RESNET_FLOWER_CHANGES = [
    {},
    {"weight_decay": 0.05},
    {"label_smoothing": 0.1},
    {"label_smoothing": 0.2},
    {"weight_decay": 0.05, "label_smoothing": 0.1},
    {"weight_decay": 0.05, "label_smoothing": 0.2},
]


def get_runs(problem):
    if problem == 1:
        return [("cnn", {**CNN_BASE, **change}) for change in CNN_CHANGES]
    if problem == 2:
        return [("resnet18", RESNET), ("resnet50", RESNET)]
    if problem == 3:
        return [("fcn", models.FCN_BEST), ("cnn", models.CNN_BEST), ("resnet18", RESNET)]
    fcn = [("fcn", {**models.FCN_BEST, **change}) for change in FCN_FLOWER_CHANGES]
    cnn = [("cnn", {**models.CNN_BEST, **change}) for change in CNN_FLOWER_CHANGES]
    resnet = [("resnet18", change) for change in RESNET_FLOWER_CHANGES]
    return fcn + cnn + resnet


parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=int, required=True, choices=[1, 2, 3, 4])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--pretrained", action="store_true")
args = parser.parse_args()

task = f"problem{args.problem}_pretrained" if args.pretrained else f"problem{args.problem}"
dataset = data.PROBLEMS[args.problem]
log = utils.get_log(utils.LOGS / f"{task}_train.txt")
log(f"{task}: {dataset} on {utils.DEVICE} | max epochs {args.epochs} | batch size {args.batch_size} | learning rate {args.lr} | patience {args.patience}")
for folder in (utils.FIGURES, utils.MODELS):
    (folder / f"{task}_configs").mkdir(exist_ok=True)

train_loader = data.get_loader(dataset, "train", args.batch_size)
val_loader = data.get_loader(dataset, "val", args.batch_size)
best = {}
runs = [("resnet18", PRETRAINED_RESNET)] if args.pretrained else get_runs(args.problem)
for name, config in runs:
    tag = "_".join([name] + [f"{key}={value}" for key, value in config.items()])
    log(f"{name} {config}")
    torch.manual_seed(0)
    model = models.get_model(name, dataset, config).to(utils.DEVICE)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    results = utils.train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, args.epochs, log, args.patience)
    _, val_acc = utils.test_step(model, val_loader, loss_fn)
    log(f"  training time {results['time']:.1f}s | epochs run {len(results['train_loss'])} | parameters {utils.count_params(model):,} | validation accuracy {val_acc:.2f}")
    title = utils.get_title(name, config)
    utils.plot_loss_curves(results, title, utils.FIGURES / f"{task}_configs" / f"{tag}.png")
    checkpoint = {"model": name, "config": config, "state": model.state_dict(), "train_time": results["time"], "epochs": len(results["train_loss"]), "val_acc": val_acc}
    torch.save(checkpoint, utils.MODELS / f"{task}_configs" / f"{tag}.pt")
    if name not in best or val_acc > best[name]["checkpoint"]["val_acc"]:
        best[name] = {"checkpoint": checkpoint, "results": results, "title": title, "summary": utils.get_summary(model, dataset)}

summaries = []
for name, record in best.items():
    checkpoint = record["checkpoint"]
    log(f"best {name}: {checkpoint['config']} | validation accuracy {checkpoint['val_acc']:.2f}")
    torch.save(checkpoint, utils.MODELS / f"{task}_{name}.pt")
    utils.plot_loss_curves(record["results"], record["title"], utils.FIGURES / f"{task}_{name}_loss.png")
    summaries.append(f"{record['title']}\n{record['summary']}")
(utils.LOGS / f"{task}_model_summary.txt").write_text("\n\n".join(summaries), encoding="utf-8")
