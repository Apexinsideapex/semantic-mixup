import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchvision import transforms, datasets
import os
import random

class SemCutMix:
    def __init__(self, model, alpha, prob, threshold):
        self.model = model
        self.alpha = alpha
        self.prob = prob
        self.threshold = threshold

    def __call__(self, batch):
        images, labels = batch
        if np.random.rand() > self.prob:
            return images, labels
        
        batch_size = len(images)
        rand_index = torch.randperm(batch_size)
        print(rand_index)
        mixed_images = images.clone()
        
        for i in range(batch_size):
            thresh = np.random.random_sample(size=None)
            bbox1 = get_bbox(images[i], thresh, self.model)
            bbox2 = get_bbox(images[rand_index[i]], thresh, self.model)

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # # Display original image
            # ax1.imshow(images[i].permute(1, 2, 0))
            # ax1.set_title(f'Original Image (Class: {dataset.classes[labels[i]]})')
            # ax1.axis('off')
            
            # # Display mixed image
            # ax2.imshow(images[rand_index[i]].permute(1, 2, 0))
            # ax2.set_title(f'SemCutMix Image (λ =)')
            # ax2.axis('off')
            
            # plt.tight_layout()
            # plt.show()
            
            x1, y1, x2, y2 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            width1, height1 = x2 - x1, y2 - y1
            width2, height2 = x2_2 - x1_2, y2_2 - y1_2
            
            if width1 > width2 or height1 > height2:
                scale = min(width2 / width1, height2 / height1)
                new_width, new_height = int(width1 * scale), int(height1 * scale)
                x2, y2 = x1 + new_width, y1 + new_height
            
            mixed_images[i][:, y1_2:y1_2+y2-y1, x1_2:x1_2+x2-x1] = images[rand_index[i]][:, y1:y2, x1:x2]
            
            bbox_area = (x2 - x1) * (y2 - y1)
            total_area = images[i].shape[1] * images[i].shape[2]
            lam = bbox_area / total_area

            # mixed_images[i] = lam * mixed_images[i] + (1 - lam) * images[rand_index[i]]

            
        return mixed_images, (labels, labels[rand_index], lam)

def get_bbox(img, threshold, model):
    input_tensor = normalize(resize(img, (224, 224)), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    with SmoothGradCAMpp(model, model.target_layer) as cam_extractor:
        out = model(input_tensor.unsqueeze(0))
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    
    mask = to_pil_image(activation_map[0].squeeze(0), mode='F')
    overlay = mask.resize((img.shape[2], img.shape[1]))
    overlay = np.array(overlay)
    overlay[overlay > threshold] = 1
    overlay[overlay <= threshold] = 0
    
    indices = np.where(overlay == 1)
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        return 0, 0, img.shape[2], img.shape[1]
    
    min_y, min_x = np.min(indices, axis=1)
    max_y, max_x = np.max(indices, axis=1)
    
    return min_x, min_y, max_x, max_y

def initialize_model(model_name, dataset_name):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.target_layer = 'layer4'
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.target_layer = 'features'
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
    
    model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models/best_models/{model_name}_{dataset_name}_best.pth'
    model.load_state_dict(torch.load(model_path))
    return model

def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img)

# Main execution
if __name__ == "__main__":
    model_name = 'resnet18'
    dataset_name = 'cub200'
    
    model = initialize_model(model_name, dataset_name)
    model.eval()

    # Set up SemCutMix
    semcutmix = SemCutMix(model, alpha=1.0, prob=1.0, threshold=0.7)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder('/home/lunet/cors13/Final_Diss/CUB_200_2011_split/train', transform=transform)
    
    # Select 10 random images from the dataset
    num_samples = 64
    indices = random.sample(range(len(dataset)), num_samples)
    images, labels = zip(*[dataset[i] for i in indices])
    images = torch.stack(images)
    labels = torch.tensor(labels)
    # Apply SemCutMix
    mixed_images, (_, _, lam) = semcutmix((images, labels))

    # Create a directory to save results
    os.makedirs('semcutmix_results', exist_ok=True)
    count = 0
    # Visualize and save results
    for i, (orig, mixed) in enumerate(zip(images, mixed_images)):
        if count == 10:
            break
        count += 1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Denormalize images for visualization
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return torch.clamp(tensor * std + mean, 0, 1)

        orig_img = denormalize(orig).permute(1, 2, 0).numpy()
        mixed_img = denormalize(mixed).permute(1, 2, 0).numpy()
        
        ax1.imshow(orig_img)
        ax1.set_title(f'Original Image (Class: {dataset.classes[labels[i]]})')
        ax1.axis('off')
        
        ax2.imshow(mixed_img)
        ax2.set_title(f'SemCutMix Image (λ = {lam:.2f})')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'test/semcutmix_result_{i}.png')
        plt.close()

    print("SemCutMix results have been saved in the 'test' directory.")