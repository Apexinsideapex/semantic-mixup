import os

class Config:
    # General
    SEED = 42
    DEVICE = "cuda"
    
    # Data
    DATA_DIR = "path/to/data"
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    
    # Model
    MODELS = [ "resnet18", "vgg16"]
    # MODELS = ['efficientnetv2', 'resnet50']
    DATASETS = [ "cub200", "stanford_dogs", "cifar10"]
    # MODELS = ["resnet18"]
    # DATASETS = ["cub200"]
    # DATASETS = ["cifar10"]
    
    # Training
    EPOCHS = 150
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    WARM_UP_EPOCHS = 50
    
    # Logging
    WANDB_PROJECT = "semantic-mixup"
    LOG_INTERVAL = 10

    #Experiements
    EXPERIMENT_DIR = "./final_experiments_base_64"

    # Augmentation
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0
    CUTMIX_PROB = 0.5

    USE_SEMCUTMIX = False
    SEMCUTMIX_ALPHA = 1.0
    SEMCUTMIX_PROB = 0.5
    SEMCUTMIX_THRESHOLD = 0.7