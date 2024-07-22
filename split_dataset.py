import os
import shutil
import random
from tqdm import tqdm
import torchvision
import torch
from PIL import Image
import numpy as np

def create_train_test_split(source_dir, dest_dir, train_ratio=0.8):
    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all class folders
    class_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    for class_folder in tqdm(class_folders, desc="Processing classes"):
        # Create class folders in train and test directories
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)

        # Get all images in the class folder
        images = [f for f in os.listdir(os.path.join(source_dir, class_folder)) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle the images
        random.shuffle(images)

        # Split images into train and test sets
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Copy images to train folder
        for img in train_images:
            src = os.path.join(source_dir, class_folder, img)
            dst = os.path.join(train_dir, class_folder, img)
            shutil.copy2(src, dst)

        # Copy images to test folder
        for img in test_images:
            src = os.path.join(source_dir, class_folder, img)
            dst = os.path.join(test_dir, class_folder, img)
            shutil.copy2(src, dst)

def prepare_cifar10(dest_dir, train_ratio=0.8):
    # Download CIFAR-10
    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Combine train and test sets
    data = np.concatenate((cifar10_train.data, cifar10_test.data), axis=0)
    targets = cifar10_train.targets + cifar10_test.targets

    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create class folders
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Shuffle the data
    combined = list(zip(data, targets))
    random.shuffle(combined)
    data, targets = zip(*combined)
    data = np.array(data)

    # Split data
    split_index = int(len(data) * train_ratio)

    # Save images
    for i, (img, label) in enumerate(tqdm(zip(data, targets), total=len(data), desc="Processing CIFAR-10")):
        img = Image.fromarray(img)
        class_name = classes[label]
        if i < split_index:
            img.save(os.path.join(train_dir, class_name, f"{i}.png"))
        else:
            img.save(os.path.join(test_dir, class_name, f"{i}.png"))

if __name__ == "__main__":
    # CUB_200_2011
    cub_source_dir = "/home/lunet/cors13/Final_Diss/CUB_200_2011/images"
    # cub_dest_dir = "/home/lunet/cors13/Final_Diss/CUB_200_2011_split"
    # create_train_test_split(cub_source_dir, cub_dest_dir)
    # print("CUB_200_2011 dataset split completed.")

    # # Stanford Dogs
    # dogs_source_dir = "/home/lunet/cors13/Final_Diss/Stanford_Dogs/Images"
    # dogs_dest_dir = "/home/lunet/cors13/Final_Diss/Stanford_Dogs_split"
    # create_train_test_split(dogs_source_dir, dogs_dest_dir)
    # print("Stanford Dogs dataset split completed.")

    # CIFAR-10
    cifar10_dest_dir = "/home/lunet/cors13/Final_Diss/CIFAR10_split"
    prepare_cifar10(cifar10_dest_dir)
    print("CIFAR-10 dataset split completed.")