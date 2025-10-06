import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models
from typing import Dict, Tuple
from PIL import Image
from tqdm import tqdm

##Considering the problem as a regression problem.

# ===========================
# Configuration
# ===========================
BASE_DIR = r"D:\braineye"
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 8
LR = 1e-3
STEP_SIZE = 7
GAMMA = 0.1


# ===========================
# Custom Dataset
# ===========================

class AgeFolderDataset(Dataset):
    """
    Custom dataset that interprets folder names as numeric age labels.
    Directory structure:
        data/
            20/   -> images of age 20
            45/   -> images of age 45
    """
    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.transform = transform

        # Loop through each folder (each representing an age)
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            try:
                age = float(folder_name)  # folder name is the label
            except ValueError:
                print(f"Skipping non-numeric folder: {folder_name}")
                continue

            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(folder_path, img_name), age))

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)
    

# ===========================
# Model
# ===========================

def create_model() -> nn.Module:
    """Load MobileNetV2 and adapt it for single-value regression."""
    model = models.mobilenet_v2(weights="DEFAULT")
    for param in model.features.parameters():
        param.requires_grad = False  # freeze backbone

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)  # single regression output
    return model

#===========================
# Sampler to handle class imbalance
#===========================

def create_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to handle class imbalance."""
    # Extract all target ages from the dataset
    ages = [int(age) for _, age in dataset.samples]

    # Count how many samples per unique age
    unique_ages, class_counts = np.unique(ages, return_counts=True)

    # Compute weight for each class (inverse of frequency)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its age
    sample_weights = np.array([class_weights[np.where(unique_ages == age)[0][0]] for age in ages])

    # Convert to tensor
    sample_weights = torch.from_numpy(sample_weights).float()

    # Create sampler
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    return sampler

# ===========================
# Create the dataloaders
# ===========================
def create_dataloaders(data_dir: str) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    
    """Create dataloaders for regression."""
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    datasets_ = {
        x: AgeFolderDataset(os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ["train", "val"]
    }

    dataloaders = {
        x: DataLoader(datasets_[x], batch_size=BATCH_SIZE,sampler=
                      create_sampler(datasets_[x]), num_workers=NUM_WORKERS)
        for x in ["train", "val"]
    }

    # dataloaders = {
    #     x: DataLoader(datasets_[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    #     for x in ["train", "val"]
    # }

    dataset_sizes = {x: len(datasets_[x]) for x in ["train", "val"]}
    return dataloaders, dataset_sizes

# ===========================
# Training Loop
# ===========================
def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    num_epochs: int = 25,
    model_dir: str = MODEL_DIR
) -> nn.Module:
    """Train and evaluate a regression model, saving the best checkpoint."""
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = os.path.join(model_dir, "age_prediction_0.0.1.pt")

    best_loss = float("inf")
    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            print(f"{phase.capitalize()} Phase")
            model.train(phase == "train")
            running_loss = 0.0

            # for inputs, targets in dataloaders[phase]:
            with tqdm(dataloaders[phase], desc=f"{phase.upper()} Epoch", unit="batch") as progress_bar:
                for inputs, targets in progress_bar:
                    inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        if phase == "val":
                            print(f"Outputs: {outputs.squeeze().cpu().numpy()}, Targets: {targets.squeeze().cpu().numpy()}")
                        loss = criterion(outputs, targets)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), best_model_path)

    elapsed = time.time() - since
    print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best Validation Loss: {best_loss:.4f}")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, dataset_sizes = create_dataloaders(DATA_DIR)
    #Validate that the dataloader are using balaced sampling
    loader = dataloaders['train']
    data_iterator = iter(loader)
    # Retrieve the first batch
    inputs, ages = next(data_iterator)
    age_counts = {}
    for age in ages.numpy():
        age = int(age)
        if age in age_counts:
            age_counts[age] += 1
        else:
            age_counts[age] = 1
    print(f"Age distribution in a batch")
    print(age_counts)

    model = create_model().to(device)

    # Regression = MSE loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
    )