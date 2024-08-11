import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F
from PIL import Image
import numpy as np
from config import Config
import hashlib

seed = Config.SEED


class CutMix(object):
    def __init__(self, alpha, prob):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        images, labels = batch
        if np.random.rand() > self.prob:
            return images, labels
        
        batch_size = len(images)
        rand_index = torch.randperm(batch_size)

        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        for i in range(batch_size):
            image = images[i].permute(1, 2, 0).cpu().numpy()
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            plt.imsave(f'./test/cutmix_{i}.png', image)

        return images, (labels, labels[rand_index], lam)
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


def initialize_model(model_name, dataset_name, use_cutmix=False):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.target_layer = 'layer4'
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(weights='DEFAULT')
        model.target_layer = 'layer4'
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(weights='DEFAULT')
        model.target_layer = 'features'
    elif model_name == 'vgg19':
        model = torchvision.models.vgg19(weights='DEFAULT')
        model.target_layer = 'features'
    elif model_name == 'efficientnetv2':
        model = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
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
    
    if model_name in ['resnet18', 'resnet50']:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name in ['vgg16', 'vgg19']:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnetv2':
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    
    model.eval()
    
    # Load best model weights
    # if use_cutmix:
    #     model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models_cutmix/best_models/{model_name}_{dataset_name}_cutmix_best.pth'
    # else:
    #     model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models/best_models/{model_name}_{dataset_name}_best.pth'
    if model_name in ['resnet18', 'vgg16']:
        # model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models_64/best_models/{model_name}_{dataset_name}_base_64_best.pth'
        model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models_cutmix/best_models/{model_name}_{dataset_name}_cutmix_best.pth'
    else:
        # model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models_b64/best_models/{model_name}_{dataset_name}_new_b64_best.pth'
        model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/cutmix_models_b64/best_models/{model_name}_{dataset_name}_new_cutmix_b64_best.pth'

    model.load_state_dict(torch.load(model_path))
    return model




class SemMixUp:
    def __init__(self, model, alpha, prob, threshold, dataset_name, save_dir='test_semmixed_images'):
        self.model = initialize_model(model_name=model, dataset_name=dataset_name, use_cutmix=False).to('cuda')
        self.alpha = alpha
        self.prob = prob
        self.threshold = threshold
        self.cam_extractor = SmoothGradCAMpp(self.model, self.model.target_layer)
        self.cached_bboxes = {}
        self.save_dir = save_dir
        self.image_count = 0
        self.rng = np.random.RandomState(seed)
        self.call_count = 0
        
    def __call__(self, batch_idx, batch):
        images, labels = batch
        rand_num = self.rng.rand()
        if rand_num > self.prob:
            return images, labels

        batch_size = len(images)
        rand_index = torch.from_numpy(self.rng.permutation(batch_size))
        mixed_images = images.clone()
        
        # Get all bboxes for the batch at once
        if batch_idx not in self.cached_bboxes.keys():
            # print(f"Generating bbox for {batch_idx}")
            overlays = self.get_batch_maps(batch_idx, images)[0]
        else:
            # print(f"Not generating bbox for {batch_idx}")
            overlays = self.cached_bboxes[batch_idx][0]

        # print(overlays)
        # overlays = torch.tensor(overlays)
        # print(overlays)
        overlays = (overlays > 0.3).float()
        overlays2 = 1 - overlays

        # overlays.to('cuda')
        # overlays2.to('cuda')
        # mixed_images.to('cuda')
        # images.to('cuda')
        

        mixed_images = mixed_images.to('cuda') * overlays.to('cuda') + images[rand_index].to('cuda') * overlays2.to('cuda')


        # self.save_mixed_images(mixed_images)
        # raise ValueError("Not implemented")
        return mixed_images, (labels, labels[rand_index], [0.5] * batch_size)
    
    def save_mixed_images(self, mixed_images):
        for i, img in enumerate(mixed_images):
            save_path = os.path.join(self.save_dir, f'semmixed_image_{self.image_count + i}.png')
            save_image(img, save_path)
        self.image_count += len(mixed_images)

    def get_batch_maps(self, batch_idx, images):
        batch_size = len(images)
        bboxes = []
        for i in range(batch_size):
            bbox = self.get_bbox(images[i])
            bboxes.append(bbox)
        self.cached_bboxes[batch_idx] = bboxes
        return bboxes
    
    def get_bbox(self, img):
        input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to('cuda')
        

        out = self.model(input_tensor.unsqueeze(0))
        activation_map = self.cam_extractor(out.squeeze(0).argmax().item(), out)
        
        if isinstance(activation_map, list):
            activation_map = activation_map[0]  # Assume the first element is the tensor we want
    
    # Ensure activation_map is a 4D tensor
        if activation_map.dim() == 3:
            activation_map = activation_map.unsqueeze(0)
        elif activation_map.dim() == 2:
            activation_map = activation_map.unsqueeze(0).unsqueeze(0)
        
        overlay = F.interpolate(activation_map, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
        
        
        return overlay




def save_image(tensor, filename, normalize=True):
    """
    Save a PyTorch tensor as an image file.
    
    Args:
    tensor (torch.Tensor): Image tensor to save. Expected shape: (C, H, W)
    filename (str): File path to save the image
    normalize (bool): If True, normalize the tensor to 0-1 range before saving
    """
    # Ensure the tensor is on CPU
    tensor = tensor.cpu().clone()
    
    # Remove the batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Normalize to 0-1 range if requested
    if normalize:
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max()
    
    # Convert to PIL Image
    if tensor.shape[0] == 1:  # Grayscale
        tensor = tensor.squeeze(0)
        image = transforms.ToPILImage()(tensor)
    else:  # RGB
        image = transforms.ToPILImage()(tensor)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the image
    image.save(filename)


class SemCutMix:
    def __init__(self, model, alpha, prob, threshold, dataset_name, save_dir='test_mixed_images'):
        self.model = initialize_model(model_name=model, dataset_name=dataset_name, use_cutmix=False).to('cuda')
        self.alpha = alpha
        self.prob = prob
        self.threshold = threshold
        self.cam_extractor = SmoothGradCAMpp(self.model, self.model.target_layer)
        self.cached_bboxes = {}
        self.save_dir = save_dir
        self.image_count = 0
        self.rng = np.random.RandomState(seed)
        self.call_count = 0
        
    def __call__(self, batch_idx, batch):
        images, labels = batch
        rand_num = self.rng.rand()
        if rand_num > self.prob:
            # print("Not using SemCutMix for this ", str(rand_num))
            return images, labels
        # print("Using SemCutMix for this ", str(rand_num))
        batch_size = len(images)
        rand_index = torch.from_numpy(self.rng.permutation(batch_size))
        mixed_images = images.clone()
        
        # Get all bboxes for the batch at once
        if batch_idx not in self.cached_bboxes.keys():
            # print(f"Generating bbox for {batch_idx}")
            bboxes1 = self.get_batch_bboxes(batch_idx, images)
        else:
            # print(f"Not generating bbox for {batch_idx}")
            bboxes1 = self.cached_bboxes[batch_idx]
        # bboxes2 = self.get_batch_bboxes(images[rand_index])
        bboxes2 = [self.cached_bboxes[batch_idx][idx] for idx in rand_index]
        # print(bboxes1)
        # print(bboxes2)
        # raise ValueError("Not implemented")
        # Vectorize the mixing operation
        # masks = torch.ones_like(images)
        # masks2 = torch.ones_like(images)
        # for i in range(batch_size):
        #     x1, y1, x2, y2 = bboxes1[i]
        #     masks[i, :, y1:y2, x1:x2] = 0
        # for i in range(bboxes2):
        #     x1, y1, x2, y2 = bboxes2[i]
        #     masks2[i, :, y1:y2, x1:x2] = 0
        

        # mixed_images = mixed_images * masks + images[rand_index] * (1 - masks2)
        lams = []
        for i in range(batch_size):
            x1_1, y1_1, x2_1, y2_1 = bboxes1[i]
            x1_2, y1_2, x2_2, y2_2 = bboxes2[i]

            # Extract content from bbox1 in image1
            content_bbox1 = images[i, :, y1_1:y2_1, x1_1:x2_1]

            # Ensure the content fits into bbox2
            content_bbox1_resized = torch.nn.functional.interpolate(content_bbox1.unsqueeze(0), size=(y2_2 - y1_2, x2_2 - x1_2), mode='bilinear', align_corners=False).squeeze(0)

            # Place the content into bbox2 in image2
            mixed_images[rand_index[i], :, y1_2:y2_2, x1_2:x2_2] = content_bbox1_resized

            # Calculate lambda
            lam = ((x2_1 - x1_1) * (y2_1 - y1_1)) / (224 * 224)
            lams.append(lam)

        # self.save_mixed_images(mixed_images)
        # raise ValueError("Not implemented")
        return mixed_images, (labels, labels[rand_index], lams)
    
    def save_mixed_images(self, mixed_images):
        for i, img in enumerate(mixed_images):
            save_path = os.path.join(self.save_dir, f'mixed_image_{self.image_count + i}.png')
            save_image(img, save_path)
        self.image_count += len(mixed_images)

    def get_batch_bboxes(self, batch_idx, images):
        batch_size = len(images)
        bboxes = []
        for i in range(batch_size):
            bbox = self.get_bbox(images[i])
            bboxes.append(bbox)
        self.cached_bboxes[batch_idx] = bboxes
        return bboxes
    
    def get_bbox(self, img):
        input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to('cuda')
        

        out = self.model(input_tensor.unsqueeze(0))
        activation_map = self.cam_extractor(out.squeeze(0).argmax().item(), out)
        
        if isinstance(activation_map, list):
            activation_map = activation_map[0]  # Assume the first element is the tensor we want
    
    # Ensure activation_map is a 4D tensor
        if activation_map.dim() == 3:
            activation_map = activation_map.unsqueeze(0)
        elif activation_map.dim() == 2:
            activation_map = activation_map.unsqueeze(0).unsqueeze(0)
        
        overlay = F.interpolate(activation_map, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
        overlay = (overlay > self.threshold).float()
        
        indices = torch.nonzero(overlay.squeeze())
        if indices.numel() == 0:
            return 0, 0, img.shape[2], img.shape[1]
        
        min_yx, _ = torch.min(indices, dim=0)
        max_yx, _ = torch.max(indices, dim=0)
        
        return min_yx[1].item(), min_yx[0].item(), max_yx[1].item(), max_yx[0].item()


class SemCutMixLoader:
    def __init__(self, loader, model, alpha, prob, threshold, dataset_name):
        self.loader = loader
        self.semcutmix = SemMixUp(model, alpha, prob, threshold, dataset_name)

    def __iter__(self):
        for batch_idx, batch in enumerate(self.loader):
            yield self.semcutmix(batch_idx, batch)

    def __len__(self):
        return len(self.loader)
class DatasetFactory:
    @staticmethod
    def get_dataset(name, train=True):
        if name == "cub200":
            return datasets.ImageFolder(
                root=os.path.join("/home/lunet/cors13/Final_Diss/CUB_200_2011_split", "train" if train else "test"),
                transform=DatasetFactory.get_transform(train, name)
            )
        elif name == "stanford_dogs":
            return datasets.ImageFolder(
                root=os.path.join("/home/lunet/cors13/Final_Diss/Stanford_Dogs_split", "train" if train else "test"),
                transform=DatasetFactory.get_transform(train, name)
            )
        elif name == "cifar10":
            return datasets.ImageFolder(
                root=os.path.join("/home/lunet/cors13/Final_Diss/CIFAR10_split", "train" if train else "test"),
                transform=DatasetFactory.get_transform(train, name)
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

    @staticmethod
    def get_transform(train, dataset_name):
        if dataset_name == "cifar10":
            if train:
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:  # For CUB200 and Stanford Dogs
            if train:
                return transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

def get_dataloader_cutmix(dataset, batch_size, num_workers, shuffle=True, use_cutmix=False, cutmix_alpha=1.0, cutmix_prob=0.5):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    
    if use_cutmix:
        return CutMixLoader(loader, cutmix_alpha, cutmix_prob)

    return loader

def get_dataloader_semcutmix(dataset, batch_size, num_workers, shuffle=True, model=None, use_semcutmix=False, semcutmix_alpha=1.0, semcutmix_prob=0.5, semcutmix_threshold=0.5, dataset_name=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    
    if use_semcutmix and model is not None:
        return SemCutMixLoader(loader, model, semcutmix_alpha, semcutmix_prob, semcutmix_threshold, dataset_name)

    return loader

class CutMixLoader:
    def __init__(self, loader, alpha, prob):
        self.loader = loader
        self.cutmix = CutMix(alpha, prob)

    def __iter__(self):
        for batch in self.loader:
            yield self.cutmix(batch)

    def __len__(self):
        return len(self.loader)