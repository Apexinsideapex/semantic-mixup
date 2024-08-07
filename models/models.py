import torch
import torch.nn as nn
from torchvision import models


class ModelFactory:
    @staticmethod
    def get_model(name, num_classes):
        if name == "resnet18":
            model = models.resnet18(weights='DEFAULT')
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == "resnet50":
            model = models.resnet50(weights='DEFAULT')
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == "vgg16":
            model = models.vgg16(weights='DEFAULT')
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif name == "vgg19":
            model = models.vgg19(weights='DEFAULT')
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif name == "efficientnetv2":
            model = models.efficientnet_v2_s(weights='DEFAULT')
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model
    
    def load_trained_model(model_name, num_classes, model_path):
        model = ModelFactory.get_model(model_name, num_classes)
        model.load_state_dict(torch.load(model_path))
        return model