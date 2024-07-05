import os

import h5py
import torch
from matplotlib import pyplot as plt


# Function to list keys in an HDF5 file
def list_hdf5_keys(file_path):
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
    return keys

def testing():
    # Inspect keys in the HDF5 files
    image_file = '/big_vesicle_cls/bigV_cls_im.h5'
    label_file = '/big_vesicle_cls/bigV_cls_label.h5'
    mask_file = '/big_vesicle_cls/bigV_cls_mask.h5'

    image_keys = list_hdf5_keys(image_file)
    label_keys = list_hdf5_keys(label_file)
    mask_keys = list_hdf5_keys(mask_file)

    print("Image file keys:", image_keys)
    print("Label file keys:", label_keys)
    print("Mask file keys:", mask_keys)

from sklearn.metrics import accuracy_score

def print_predictions_and_compare_accuracy(model, data_loader, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            inputs = inputs.float().unsqueeze(1).to(device)  # Add channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Collect true labels and predictions
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Print predictions and labels for the current batch
            print(f"Batch {batch_idx}:")
            print(f"Predictions: {predicted.cpu().numpy()}")
            print(f"Labels: {targets.cpu().numpy()}")

    # Calculate and print overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Overall Accuracy: {accuracy:.4f}')
