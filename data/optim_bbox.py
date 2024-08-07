import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
import time
import numpy as np
from PIL import Image
import os
from glob import glob

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model
model = models.resnet18(pretrained=True).to(device)
target_layer = model.layer4[-1]

# Initialize GradCAM
cam = GradCAM(model=model, target_layers=[target_layer])

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_bbox(activation_map, threshold=0.5):
    overlay = F.interpolate(activation_map, size=(224, 224), mode='bilinear', align_corners=False)
    overlay = (overlay > threshold).float()
    
    indices = torch.nonzero(overlay.squeeze())
    if indices.numel() == 0:
        return 0, 0, 224, 224
    
    min_yx, _ = torch.min(indices, dim=0)
    max_yx, _ = torch.max(indices, dim=0)
    
    return min_yx[1].item(), min_yx[0].item(), max_yx[1].item(), max_yx[0].item()

def process_batch(images, target_category=None):
    # Preprocess images
    input_tensor = torch.stack(images).to(device)
    
    # Generate class activation maps
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    
    # Convert to torch tensor
    activation_maps = torch.from_numpy(grayscale_cam).to(device)
    
    # Get bounding boxes
    bboxes = [get_bbox(activation_map.unsqueeze(0).unsqueeze(0)) for activation_map in activation_maps]
    
    return bboxes

def load_images(directory, batch_size):
    image_paths = glob(os.path.join(directory, '*', '*.jpg'))
    images = []
    for path in image_paths[:batch_size]:
        img = Image.open(path).convert('RGB')
        img_tensor = preprocess(img)
        images.append(img_tensor)
    return images

def test_batch_processing(directory, batch_size=32, num_runs=10):
    # Load images
    images = load_images(directory, batch_size)

    # Warm-up run
    _ = process_batch(images)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        bboxes = process_batch(images)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Average time for batch of {batch_size}: {avg_time:.4f} ± {std_time:.4f} seconds")
    return bboxes, avg_time

if __name__ == "__main__":
    directory = "/home/lunet/cors13/CUB200_split/train"
    bboxes, avg_time = test_batch_processing(directory, batch_size=32, num_runs=10)
    print(f"Sample bounding boxes: {bboxes[:5]}")