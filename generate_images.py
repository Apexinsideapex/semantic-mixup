import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from data.datasets import DatasetFactory, CutMix, SemCutMix, SemMixUp

def save_image_pair(img1, img2, title1, title2, caption, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))  # Increased height
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax1.axis('off')
    ax2.imshow(img2)
    ax2.set_title(title2)
    ax2.axis('off')
    plt.suptitle(caption, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust top margin
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_image_row(images, titles, caption, filename):
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 6))  # Increased height
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle(caption, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust top margin
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def tensor_to_pil(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor.cpu().clone().squeeze(0))

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def main():
    datasets = ["cub200", "stanford_dogs", "cifar10"]
    output_dir = "augmentation_examples"
    os.makedirs(output_dir, exist_ok=True)
    num_samples = 3  # Number of samples to process for each dataset

    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        train_dataset = DatasetFactory.get_dataset(dataset_name, train=True)
        
        if dataset_name == "cifar10":
            mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        cutmix = CutMix(alpha=1.0, prob=np.inf)
        semcutmix = SemCutMix("resnet18", alpha=1.0, prob=np.inf, threshold=0.5, dataset_name=dataset_name)
        semmixup = SemMixUp("resnet18", alpha=1.0, prob=np.inf, threshold=0.5, dataset_name=dataset_name)

        for sample in range(num_samples):
            idx1, idx2 = np.random.choice(len(train_dataset), 2, replace=False)
            img1, label1 = train_dataset[idx1]
            img2, label2 = train_dataset[idx2]

            img1 = denormalize(img1, mean, std)
            img2 = denormalize(img2, mean, std)

            batch = torch.stack([img1, img2]), torch.tensor([label1, label2])

            cutmix_batch, _ = cutmix(batch)
            cutmix_img = cutmix_batch[0]

            semcutmix_batch, _ = semcutmix(0, batch)
            semcutmix_img = semcutmix_batch[0]

            semmixup_batch, _ = semmixup(0, batch)
            semmixup_img = semmixup_batch[0]

            img1_pil = tensor_to_pil(img1)
            cutmix_pil = tensor_to_pil(cutmix_img)
            semcutmix_pil = tensor_to_pil(semcutmix_img)
            semmixup_pil = tensor_to_pil(semmixup_img)

            # Save individual augmentation pairs
            save_image_pair(
                img1_pil, cutmix_pil, "Base", "CutMix",
                f"CutMix Augmentation - {dataset_name.upper()} Dataset",
                os.path.join(output_dir, f"{dataset_name}_cutmix_sample{sample+1}.png")
            )
            save_image_pair(
                img1_pil, semcutmix_pil, "Base", "SemCutMix",
                f"SemCutMix Augmentation - {dataset_name.upper()} Dataset",
                os.path.join(output_dir, f"{dataset_name}_semcutmix_sample{sample+1}.png")
            )
            save_image_pair(
                img1_pil, semmixup_pil, "Base", "SemMixUp",
                f"SemMixUp Augmentation - {dataset_name.upper()} Dataset",
                os.path.join(output_dir, f"{dataset_name}_semmixup_sample{sample+1}.png")
            )

            # Save combined image with all augmentations
            save_image_row(
                [img1_pil, cutmix_pil, semcutmix_pil, semmixup_pil],
                ["Base", "CutMix", "SemCutMix", "SemMixUp"],
                f"Augmentation Comparison - {dataset_name.upper()} Dataset",
                os.path.join(output_dir, f"{dataset_name}_all_augmentations_sample{sample+1}.png")
            )

        print(f"Images for {dataset_name} saved in {output_dir}")

if __name__ == "__main__":
    main()