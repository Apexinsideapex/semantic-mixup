import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from config import Config
from data.datasets import DatasetFactory, get_dataloader_cutmix, get_dataloader_semcutmix
from models.models import ModelFactory
from utils.utils import set_seed, accuracy

def train(model, train_loader, optimizer, criterion, device, use_cutmix, use_semcutmix):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # Hard code GPU
    device = torch.device("cuda")
    use_cutmix = Config.USE_CUTMIX
    use_semcutmix = Config.USE_SEMCUTMIX
    for batch_idx, batch in enumerate(train_loader):
        if use_cutmix:
            inputs, targets = batch
            inputs = inputs.to(device)
            if isinstance(targets, tuple):
                targets_a, targets_b, lam = targets
                targets_a, targets_b = targets_a.to(device), targets_b.to(device)
            else:
                targets = targets.to(device)
        elif use_semcutmix:
            inputs, targets = batch
            inputs = inputs.to(device)
            if isinstance(targets, tuple):
                targets_a, targets_b, lam = targets
                targets_a, targets_b = targets_a.to(device), targets_b.to(device)
        else:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # targets = targets.to(device)
        # print(outputs.get_device())
        # print(targets.get_device())
        
        if use_cutmix and isinstance(targets, tuple):
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif use_semcutmix and isinstance(targets, tuple):
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        if use_cutmix and isinstance(targets, tuple):
            total += targets_a.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float()
                        + (1 - lam) * predicted.eq(targets_b).sum().float())
        elif use_semcutmix and isinstance(targets, tuple):
            total += targets_a.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float()
                        + (1 - lam) * predicted.eq(targets_b).sum().float())
        else:
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if batch_idx % Config.LOG_INTERVAL == 0:
            wandb.log({
                "train_loss": loss.item(),
                "train_acc": 100. * correct / total
            })

    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    wandb.log({"val_loss": val_loss, "val_acc": val_acc})

    return val_loss, val_acc

def main():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create a directory to save the best models
    best_models_dir = os.path.join(Config.EXPERIMENT_DIR, "best_models")
    os.makedirs(best_models_dir, exist_ok=True)

    for dataset_name in Config.DATASETS:
        for model_name in Config.MODELS:
            experiment_name = f"{model_name}_{dataset_name}_semcutmix_64_new"
            print(f"Running experiment: {experiment_name}")
            wandb.init(project=Config.WANDB_PROJECT, name=experiment_name)

            train_dataset = DatasetFactory.get_dataset(dataset_name, train=True)
            val_dataset = DatasetFactory.get_dataset(dataset_name, train=False)
            if Config.USE_CUTMIX:
                train_loader = get_dataloader_cutmix(train_dataset, Config.BATCH_SIZE, Config.NUM_WORKERS, 
                                            use_cutmix=Config.USE_CUTMIX, 
                                            cutmix_alpha=Config.CUTMIX_ALPHA, 
                                            cutmix_prob=Config.CUTMIX_PROB)
            elif Config.USE_SEMCUTMIX:
                train_loader = get_dataloader_semcutmix(train_dataset, Config.BATCH_SIZE, Config.NUM_WORKERS, 
                                            use_semcutmix=Config.USE_SEMCUTMIX, 
                                            semcutmix_alpha=Config.SEMCUTMIX_ALPHA, 
                                            semcutmix_prob=Config.SEMCUTMIX_PROB, 
                                            semcutmix_threshold=Config.SEMCUTMIX_THRESHOLD)
            else:
                train_loader = get_dataloader_cutmix(train_dataset, Config.BATCH_SIZE, Config.NUM_WORKERS, shuffle=True)
            val_loader = get_dataloader_cutmix(val_dataset, Config.BATCH_SIZE, Config.NUM_WORKERS, shuffle=False)


            num_classes = len(train_dataset.classes)
            model_path = f'/home/lunet/cors13/Final_Diss/semantic-mixup/base_models/best_models/{model_name}_{dataset_name}_best.pth'
            if Config.USE_SEMCUTMIX:
                model = ModelFactory.load_trained_model(model_name, num_classes, model_path).to(device)
            else:
                model = ModelFactory.get_model(model_name, num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)

            best_val_acc = 0.0
            early_stopping_counter = 0
            for epoch in range(Config.EPOCHS):
                train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, Config.USE_CUTMIX, Config.USE_SEMCUTMIX)
                val_loss, val_acc = validate(model, val_loader, criterion, device)

                print(f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # Save the best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = os.path.join(best_models_dir, f"{experiment_name}_best.pth")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved to {best_model_path}")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= Config.PATIENCE and epoch >= Config.WARM_UP_EPOCHS:
                        print("Early stopping triggered!")
                        break
                

            wandb.finish()

if __name__ == "__main__":
    main()