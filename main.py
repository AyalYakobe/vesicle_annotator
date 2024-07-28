import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import find_contours
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import time

from scripts.getting_started_script import load_hdf5, CustomDataset, BATCH_SIZE, Net, n_channels, n_classes, \
    create_data_loaders, create_model, train_model, load_checkpoint, task, NUM_EPOCHS
from scripts.html_visuals import generate_html, test_model, create_test_data_loaders

# File paths
image_file = '/Users/ayalyakobe/vesicle_annotations/data/big_vesicle_cls/bigV_cls_im_v2.h5'
label_file = '/Users/ayalyakobe/vesicle_annotations/data/big_vesicle_cls/bigV_cls_label_v2.h5'
mask_file = '/Users/ayalyakobe/vesicle_annotations/data/big_vesicle_cls/bigV_cls_mask_v2.h5'

test_image_file = '/Users/ayalyakobe/vesicle_annotations/data/big_vesicle_cls_testing/bigV_cls_202406_im.h5'
test_mask_file = '/Users/ayalyakobe/vesicle_annotations/data/big_vesicle_cls_testing/bigV_cls_202406_mask.h5'

CHECKPOINT_PATH = '/Users/ayalyakobe/vesicle_annotations/model_checkpoint.pth'

# Function to load model checkpoint
def load_model_checkpoint(model, optimizer, checkpoint_path):
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

# Function to train the model and save checkpoints
def train_and_save_model(image_file, label_file, mask_file, checkpoint_path):
    train_loader = create_data_loaders(image_file, label_file, mask_file, BATCH_SIZE)
    model, criterion, optimizer = create_model(n_channels, n_classes, task)
    start_epoch = load_model_checkpoint(model, optimizer, checkpoint_path)

    start_time = time.time()
    train_model(model, criterion, optimizer, train_loader, task, start_epoch=start_epoch, num_epochs=NUM_EPOCHS, checkpoint_path=checkpoint_path)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def show_visuals():
    logging.info("Custom Dataset Processing")
    save_dir = '/Users/ayalyakobe/vesicle_annotations/image_predictions'
    logging.info('==> Evaluating ...')

    start_time = time.time()
    # Create data loader for test images
    test_loader, true_labels = create_test_data_loaders(test_image_file, test_mask_file, label_file, BATCH_SIZE)
    model = Net(in_channels=n_channels, num_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    load_model_checkpoint(model, optimizer, CHECKPOINT_PATH)

    # Perform predictions on test images and evaluate
    test_model(model, test_loader, true_labels, save_dir)
    end_time = time.time()
    print(f"Visualizations generated in {end_time - start_time:.2f} seconds")

def eval_model_results(image_file, label_file, mask_file):
    model = Net(in_channels=n_channels, num_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    load_model_checkpoint(model, optimizer, CHECKPOINT_PATH)  # Use the modified function to load the model
    print("Evaluating model on validation/test data")
    val_loader = create_data_loaders(image_file, label_file, mask_file, BATCH_SIZE)  # Correct the order of arguments

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            inputs, labels, _ = data  # Correct the unpacking of the data
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure y_true and y_pred have consistent lengths
    if len(y_true) != len(y_pred):
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

if __name__ == '__main__':
    # train_and_save_model(image_file, label_file, mask_file, CHECKPOINT_PATH)
    # eval_model_results(image_file, label_file, mask_file)
    show_visuals()
