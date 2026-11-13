import torch
from torch import nn
from torchvision import models

from data import INFO, ROOT

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU}
CNN_BEST = {"layers": 9, "filters": 32, "stride": 1, "hidden": 256, "activation": "relu", "batch_norm": True, "dropout": 0.0}
FCN_BEST = {"layers": 10, "hidden": 256, "activation": "relu", "batch_norm": True, "dropout": 0.0}


def FCN(channels, size, classes, config):
    activation = ACTIVATIONS[config["activation"]]
    inputs = channels * size * size
    modules = [nn.Flatten()]
    for _ in range(config["layers"]):
        modules.append(nn.Linear(inputs, config["hidden"]))
        if config["batch_norm"]:
            modules.append(nn.BatchNorm1d(config["hidden"]))
        modules += [activation(), nn.Dropout(config["dropout"])]
        inputs = config["hidden"]
    modules.append(nn.Linear(inputs, classes))
    return nn.Sequential(*modules)


def CNN(channels, size, classes, config):
    activation = ACTIVATIONS[config["activation"]]
    modules = []
    current = channels
    for i in range(config["layers"]):
        out = config["filters"] * 2 ** (i // 2)
        stride = config["stride"] if i == 0 else 1
        modules.append(nn.Conv2d(current, out, 3, stride, 1))
        if config["batch_norm"]:
            modules.append(nn.BatchNorm2d(out))
        modules.append(activation())
        if i % 2 == 1:
            modules.append(nn.MaxPool2d(2))
        current = out
    features = nn.Sequential(*modules)
    flat = features.eval()(torch.zeros(1, channels, size, size)).numel()
    head = [nn.Flatten(), nn.Linear(flat, config["hidden"])]
    if config["batch_norm"]:
        head.append(nn.BatchNorm1d(config["hidden"]))
    head += [activation(), nn.Dropout(config["dropout"]), nn.Linear(config["hidden"], classes)]
    return nn.Sequential(features, nn.Sequential(*head))


def ResNet18(channels, classes):
    net = models.resnet18(num_classes=classes)
    net.conv1 = nn.Conv2d(channels, 64, 3, 1, 1, bias=False)
    net.maxpool = nn.Identity()
    return net


def ResNet18Pretrained(channels, classes):
    torch.hub.set_dir(str(ROOT / "pretrained"))
    net = models.resnet18(weights="IMAGENET1K_V1")
    net.conv1 = nn.Conv2d(channels, 64, 3, 1, 1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(512, classes)
    return net


def ResNet50(channels, classes):
    net = models.resnet50(num_classes=classes)
    net.conv1 = nn.Conv2d(channels, 64, 3, 1, 1, bias=False)
    net.maxpool = nn.Identity()
    return net


def get_model(name, dataset, config):
    info = INFO[dataset]
    classes = len(info["names"])
    if name == "fcn":
        return FCN(info["channels"], info["size"], classes, config)
    if name == "cnn":
        return CNN(info["channels"], info["size"], classes, config)
    if name == "resnet18" and config.get("pretrained"):
        return ResNet18Pretrained(info["channels"], classes)
    if name == "resnet18":
        return ResNet18(info["channels"], classes)
    return ResNet50(info["channels"], classes)
