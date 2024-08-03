import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp

def get_bbox(img, threshold, model):
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
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

def sem_cutmix(img1, img2, model, threshold=0.5):
    # Get bounding boxes for both images
    bbox1 = get_bbox(img1, threshold, model)
    bbox2 = get_bbox(img2, threshold, model)
    
    # Create mixed image (start with img2 as base)
    mixed_img = img2.clone()
    
    # Replace bbox area of img2 with bbox area from img1
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Ensure the bbox from img1 fits within img2
    width1, height1 = x2 - x1, y2 - y1
    width2, height2 = x2_2 - x1_2, y2_2 - y1_2
    
    # Adjust bbox1 size if it's larger than bbox2
    if width1 > width2 or height1 > height2:
        scale = min(width2 / width1, height2 / height1)
        new_width, new_height = int(width1 * scale), int(height1 * scale)
        x2, y2 = x1 + new_width, y1 + new_height
    
    # Paste bbox1 content into bbox2 location
    mixed_img[:, y1_2:y1_2+y2-y1, x1_2:x1_2+x2-x1] = img1[:, y1:y2, x1:x2]
    
    # Calculate the area of the bounding box
    bbox_area = (x2 - x1) * (y2 - y1)
    total_area = img1.shape[1] * img1.shape[2]
    
    # Calculate the mixing ratio (lambda)
    lam = bbox_area / total_area
    
    return mixed_img, (x1_2, y1_2, x1_2+(x2-x1), y1_2+(y2-y1)), lam

def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = torchvision.transforms.ToTensor()(img)
    return img

def initialize_model(model_name, dataset_name, use_cutmix=False):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18()
        model.target_layer = 'layer4'
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16()
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
# Main execution
if __name__ == "__main__":
    # Load pre-trained ResNet model
    model = initialize_model('resnet18', 'cub200')

    # Load two sample images
    img1 = load_and_preprocess_image('/home/lunet/cors13/Final_Diss/CUB_200_2011_split/train/187.American_Three_toed_Woodpecker/American_Three_Toed_Woodpecker_0012_179905.jpg')
    img2 = load_and_preprocess_image('/home/lunet/cors13/Final_Diss/CUB_200_2011_split/train/177.Prothonotary_Warbler/Prothonotary_Warbler_0046_174104.jpg')

    # Apply semantic CutMix
    mixed_img, bbox, lam = sem_cutmix(img1, img2, model)

    # Print class percentages
    print(f"Class percentages after SemCutMix:")
    print(f"Image 1: {lam*100:.2f}%")
    print(f"Image 2: {(1-lam)*100:.2f}%")

    # Visualize results
    fig, axs = plt.subplots(2, 2, figsize=(30, 30))
    axs[0, 0].imshow(img1.permute(1, 2, 0))
    axs[0, 0].set_title('Image 1')
    bbox1 = get_bbox(img1, 0.5, model)
    axs[0, 0].add_patch(plt.Rectangle((bbox1[0], bbox1[1]), bbox1[2]-bbox1[0], bbox1[3]-bbox1[1], 
                     fill=False, edgecolor='red', linewidth=2))
    
    axs[0, 1].imshow(img2.permute(1, 2, 0))
    axs[0, 1].set_title('Image 2')
    bbox2 = get_bbox(img2, 0.5, model)
    axs[0, 1].add_patch(plt.Rectangle((bbox2[0], bbox2[1]), bbox2[2]-bbox2[0], bbox2[3]-bbox2[1], 
                     fill=False, edgecolor='red', linewidth=2))
    
    axs[1, 0].imshow(mixed_img.permute(1, 2, 0))
    axs[1, 0].set_title('SemCutMix Result')
    axs[1, 0].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                     fill=False, edgecolor='red', linewidth=2))

    for ax in axs.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()