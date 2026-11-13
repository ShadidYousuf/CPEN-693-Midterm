import argparse

import torch
from torch import nn

import data
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=int, required=True, choices=[1, 2, 3, 4])
parser.add_argument("--pretrained", action="store_true")
args = parser.parse_args()

task = f"problem{args.problem}_pretrained" if args.pretrained else f"problem{args.problem}"
dataset = data.PROBLEMS[args.problem]
log = utils.get_log(utils.LOGS / f"{task}_eval.txt")
test_loader = data.get_loader(dataset, "test", 256)
loss_fn = nn.CrossEntropyLoss()


def evaluate(path):
    checkpoint = torch.load(path, map_location=utils.DEVICE)
    model = models.get_model(checkpoint["model"], dataset, checkpoint["config"]).to(utils.DEVICE)
    model.load_state_dict(checkpoint["state"])
    test_loss, test_acc = utils.test_step(model, test_loader, loss_fn)
    log(f"{checkpoint['model']} {checkpoint['config']}")
    log(f"  parameters {utils.count_params(model):,} | training time {checkpoint['train_time']:.1f}s | epochs run {checkpoint['epochs']} | validation accuracy {checkpoint['val_acc']:.2f} | test loss {test_loss:.4f} | test accuracy {test_acc:.2f}")


log(f"{task}: {dataset} test results of all configurations")
for path in sorted((utils.MODELS / f"{task}_configs").glob("*.pt")):
    evaluate(path)
log("")
log(f"{task}: {dataset} test results of the best configuration of each model (chosen by validation accuracy)")
for name in ("fcn", "cnn", "resnet18", "resnet50"):
    if (utils.MODELS / f"{task}_{name}.pt").exists():
        evaluate(utils.MODELS / f"{task}_{name}.pt")
