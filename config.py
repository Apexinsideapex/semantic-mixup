import os

class Config:
    # General
    SEED = 42
    DEVICE = "cuda"
    
    # Data
    DATA_DIR = "path/to/data"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Model
    MODELS = ["resnet18", "vgg16"]
    DATASETS = ["cub200", "stanford_dogs", "cifar10"]
    
    # Training
    EPOCHS = 150
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # Logging
    WANDB_PROJECT = "semantic-mixup"
    LOG_INTERVAL = 10

    #Experiements
    EXPERIMENT_DIR = "./base_models_cutmix"

    # Augmentation
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0
    CUTMIX_PROB = 0.5