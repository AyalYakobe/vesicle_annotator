import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import find_contours
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from scripts.getting_started_script import load_hdf5, CustomDataset, BATCH_SIZE, Net, n_channels, n_classes


# Function to visualize image, prediction, and mask
def visualize_image_with_prediction(image, prediction, mask, label, pred_label, save_dir, index):
    print(f"Image shape: {image.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Mask shape: {mask.shape}")

    num_slices = image.shape[1]  # Number of slices in the second dimension
    fig, ax = plt.subplots(2, num_slices, figsize=(20, 6))

    for i in range(num_slices):
        contours_mask = find_contours(mask[i], level=0.5)
        contours_pred = find_contours(prediction[i], level=0.5)

        if pred_label == 0:
            pred_label_text = 'DV'
            color = 'blue'
        elif pred_label == 1:
            pred_label_text = 'CV'
            color = 'green'
        else:
            pred_label_text = 'DVH'
            color = 'red'

        # Display the original image slice with mask outline
        ax[0, i].imshow(image[:, i, :], cmap='gray')
        for contour in contours_mask:
            ax[0, i].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        ax[0, i].set_title(f'Image with Mask Outline Slice {i}')
        ax[0, i].axis('off')

        # Display the original image slice with prediction outline
        ax[1, i].imshow(image[:, i, :], cmap='gray')
        for contour in contours_pred:
            ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)
        ax[1, i].set_title(f'Image with Prediction Outline Slice {i} ({pred_label_text})')
        ax[1, i].axis('off')

    plt.suptitle(f'True Label: {label}, Pred Label: {pred_label_text}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'prediction_{index}.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved prediction image to {save_path}")
    return save_path  # Return the save path of the image


def convert_to_one_hot(prediction, shape, num_classes):
    one_hot_mask = np.zeros((num_classes, *shape), dtype=np.int32)
    for i in range(num_classes):
        one_hot_mask[i, ...] = (prediction == i)
    return one_hot_mask


def create_test_data_loaders(image_file, mask_file, batch_size):
    images = load_hdf5(image_file)['main']
    masks = load_hdf5(mask_file)['main']

    dataset = CustomDataset(images=images, masks=masks, transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


# Function to perform predictions on test images
def test_model(model, data_loader, save_dir):
    model.eval()
    y_true = []
    y_pred = []
    image_paths = []  # List to store the paths of saved images
    with torch.no_grad():
        for index, (inputs, masks) in enumerate(data_loader):
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

                # Visualize the prediction and save the image
                save_path = visualize_image_with_prediction(image, prediction_one_hot[1], mask, 'N/A', pred_label, save_dir, index * len(inputs) + i + 1)
                image_paths.append(save_path)  # Collect the path of the saved image

                if len(image_paths) >= 10:  # For example, stop after saving 10 images
                    return image_paths
    return image_paths


def load_model_checkpoint(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Model checkpoint loaded from {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint)
            logging.info(f"Model state dict loaded directly from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to load model checkpoint from {checkpoint_path}: {e}")


def generate_html(image_paths, html_path):
    with open(html_path, 'w') as f:
        f.write('<html>\n<head>\n<title>Image Predictions</title>\n</head>\n<body>\n')
        for img_path in image_paths:
            relative_path = os.path.relpath(img_path, os.path.dirname(html_path))
            f.write(f'<img src="{relative_path}" alt="{os.path.basename(img_path)}"><br>\n')
        f.write('</body>\n</html>')
    print(f"Saved HTML file to {html_path}")


if __name__ == '__main__':
    import time

    start_time = time.time()

    logging.info("Custom Dataset Processing")
    image_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_im_v2.h5'
    label_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_label_v2.h5'
    mask_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls/bigV_cls_mask_v2.h5'
    test_image_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls_testing/bigV_cls_202406_im.h5'
    test_mask_file = '/Users/ayalyakobe/vesicle_annotations/big_vesicle_cls_testing/bigV_cls_202406_mask.h5'

    save_dir = '/Users/ayalyakobe/vesicle_annotations/image_predictions'
    html_path = os.path.join(save_dir, 'predictions.html')

    logging.info('==> Evaluating ...')

    # Create data loader for test images
    test_loader = create_test_data_loaders(test_image_file, test_mask_file, BATCH_SIZE)
    model = Net(in_channels=n_channels, num_classes=n_classes)
    load_model_checkpoint(model, '/Users/ayalyakobe/vesicle_annotations/model_checkpoint.pth')

    # Perform predictions on test images and evaluate
    image_paths = test_model(model, test_loader, save_dir)
    generate_html(image_paths, html_path)

    logging.info(f"Model testing completed in {time.time() - start_time:.2f} seconds")
