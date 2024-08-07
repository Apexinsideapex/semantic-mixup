import torch
import torchvision

def initialize_model(model_name, dataset_name, use_cutmix=False):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.target_layer = 'layer4'
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(weights='DEFAULT')
        model.target_layer = 'avgpool'
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if dataset_name == 'cub200':
        num_classes = 200
    elif dataset_name == 'stanford_dogs':
        num_classes = 120
    elif dataset_name == 'cifar10':
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if model_name == 'resnet18':
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg16':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    
    model.eval()
    
    # Load best model weights
    if use_cutmix:
        model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models_cutmix/best_models/{model_name}_{dataset_name}_cutmix_best.pth'
    else:
        model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models/best_models/{model_name}_{dataset_name}_best.pth'
    
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == "__main__":
    model_name = 'resnet18'
    dataset_name = 'cub200'
    
    model = initialize_model(model_name, dataset_name)

    print(list(model.named_modules()))