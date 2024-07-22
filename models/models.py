import torch
import torch.nn as nn
from torchvision import models

class ModelFactory:
    @staticmethod
    def get_model(name, num_classes):
        if name == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == "vgg16":
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model