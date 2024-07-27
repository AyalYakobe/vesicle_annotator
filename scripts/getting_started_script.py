import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize constants
NUM_EPOCHS = 20
BATCH_SIZE = 128
lr = 0.001
CHECKPOINT_PATH = '../model_checkpoint.pth'
n_channels = 1
n_classes = 3
task = 'classification'  # Specify the task type

# Function to load HDF5 files
def load_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        data = {key: np.array(f[key]) for key in f.keys()}
    return data

# Function to preprocess data
def preprocess_data(image):
    image = torch.tensor(image, dtype=torch.float32)
    mean = image.mean()
    std = image.std()
    return (image - mean) / std

# Custom dataset class
class CustomDataset(data.Dataset):
    def __init__(self, images, labels=None, masks=None, transform=None):
        self.images = images
        self.labels = labels
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] if self.labels is not None else None
        mask = self.masks[idx] if self.masks is not None else None

        # Reshape image to [5, 31, 31]
        image = image.transpose(2, 0, 1)  # from [31, 31, 5] to [5, 31, 31]

        # Normalize the image
        if self.transform:
            image = self.transform(image)
        else:
            image = preprocess_data(image)

        if label is not None:
            return image, label, mask
        else:
            return image, mask

# Function to create data loaders
def create_data_loaders(image_file, label_file, mask_file, batch_size):
    images = load_hdf5(image_file)['main']
    labels = load_hdf5(label_file)['main']-1
    masks = load_hdf5(mask_file)['main']

    dataset = CustomDataset(images, labels, masks, transform=None)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

# Define a simple CNN model
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))  # Kernel size 1 for depth
        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))  # Kernel size 1 for depth
        self.fc = nn.Sequential(
            nn.Linear(13888, 128),  # Adjusted input size based on printed shapes
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Function to create model
def create_model(in_channels, num_classes, task):
    model = Net(in_channels=in_channels, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, criterion, optimizer

# Training function with checkpointing
def train_model(model, criterion, optimizer, train_loader, task, start_epoch=0, num_epochs=NUM_EPOCHS,
                checkpoint_path=CHECKPOINT_PATH):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for inputs, targets, _ in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            targets = targets.long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path=CHECKPOINT_PATH):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0

# Evaluation function
def evaluate_model(model, data_loader, task, split, save_path=None):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Print debug information
            print(f"Batch {batch_idx}: inputs shape {inputs.shape}, predicted shape {predicted.shape}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'{split} Accuracy: {accuracy:.4f}')
    print(f'{split} Precision: {precision:.4f}')
    print(f'{split} Recall: {recall:.4f}')
    print(f'{split} F1 Score: {f1:.4f}')