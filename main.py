import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import find_contours
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils import data
from torch.utils.data import DataLoader

from scripts.getting_started_script import load_hdf5, CustomDataset, create_data_loaders, BATCH_SIZE, create_model, \
    train_model


# Function to visualize image, prediction, and mask
def visualize_image_with_prediction(image, prediction, mask, label, pred_label):
    print(f"Image shape: {image.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Mask shape: {mask.shape}")

    num_slices = image.shape[1]  # Number of slices in the second dimension
    fig, ax = plt.subplots(4, num_slices, figsize=(20, 12))

    for i in range(num_slices):
        contours_mask = find_contours(mask[i], level=0.5)
        contours_pred = find_contours(prediction[i], level=0.5)

        # Display the original image slice
        ax[0, i].imshow(image[:, i, :], cmap='gray')
        ax[0, i].set_title(f'Original Image Slice {i}')
        ax[0, i].axis('off')

        # Display the image with the mask outline slice
        ax[1, i].imshow(image[:, i, :], cmap='gray')
        for contour in contours_mask:
            ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        ax[1, i].set_title(f'Image with Mask Outline Slice {i}')
        ax[1, i].axis('off')

        # Display the image with the prediction outline slice
        ax[2, i].imshow(image[:, i, :], cmap='gray')
        for contour in contours_pred:
            ax[2, i].plot(contour[:, 1], contour[:, 0], linewidth=2, color='blue')
        ax[2, i].set_title(f'Image with Prediction Outline Slice {i}')
        ax[2, i].axis('off')

        # Display the image with the prediction overlay slice only if a vesicle is predicted
        ax[3, i].imshow(image[:, i, :], cmap='gray')
        if pred_label == 1:  # Assuming label 1 indicates the presence of a vesicle
            ax[3, i].imshow(prediction[i], cmap='Blues', alpha=0.5)  # Overlay prediction with transparency
        ax[3, i].set_title(f'True Label: {label}, Pred Label: {pred_label}')
        ax[3, i].axis('off')

    plt.suptitle(f'Label: {label}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def convert_to_one_hot(prediction, shape, num_classes):
    one_hot_mask = np.zeros((num_classes, *shape), dtype=np.int32)
    for i in range(num_classes):
        one_hot_mask[i, ...] = (prediction == i)
    return one_hot_mask


def visualize_model_predictions(model, data_loader):
    model.eval()
    with torch.no_grad():
        for inputs, targets, masks in data_loader:
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                image = inputs[i, 0].cpu().numpy()
                prediction = predicted[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                label = targets[i].cpu().numpy()
                pred_label = predicted[i].cpu().item()

                prediction_one_hot = convert_to_one_hot(prediction, mask.shape, n_classes)

                visualize_image_with_prediction(image, prediction_one_hot[1], mask, label, pred_label)


def create_test_data_loaders(image_file, mask_file, batch_size):
    images = load_hdf5(image_file)['main']
    masks = load_hdf5(mask_file)['main']

    dataset = CustomDataset(images=images, masks=masks, transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


# Function to perform predictions on test images
def test_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, masks in data_loader:
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                image = inputs[i, 0].cpu().numpy()
                prediction = predicted[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                pred_label = predicted[i].cpu().item()

                # Convert the predicted labels to one-hot encoded mask
                prediction_one_hot = convert_to_one_hot(prediction, image.shape, n_classes)

                # Collecting true and predicted labels for evaluation
                y_true.extend(mask.flatten())
                y_pred.extend(prediction.flatten())

                # Visualize the prediction
                visualize_image_with_prediction(image, prediction_one_hot[1], mask, 'N/A', pred_label)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')


if __name__ == '__main__':
    import time

    start_time = time.time()  # Start the timer

    print("Custom Dataset Processing")
    image_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_im_v2.h5'
    label_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_label_v2.h5'
    mask_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_mask_v2.h5'
    test_image_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls_testing/bigV_cls_202406_im.h5'
    test_mask_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls_testing/bigV_cls_202406_mask.h5'

    task = 'classification'  # Specify the task type
    n_channels = 1  # Number of channels for grayscale image_predictions
    n_classes = 3  # Number of classes

    # Load training data and create data loader
    train_loader = create_data_loaders(image_file, label_file, mask_file, BATCH_SIZE)

    # Create and train the model
    model, criterion, optimizer = create_model(n_channels, n_classes, task)
    train_model(model, criterion, optimizer, train_loader, task)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f'Total time taken: {elapsed_time:.2f} seconds')
    print('==> Evaluating ...')

    # Create data loader for test images
    test_loader = create_test_data_loaders(test_image_file, test_mask_file, BATCH_SIZE)

    # Perform predictions on test images and evaluate
    test_model(model, test_loader)
