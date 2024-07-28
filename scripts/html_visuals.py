import os
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import find_contours
from torch.utils.data import DataLoader

from scripts.getting_started_script import load_hdf5, CustomDataset, BATCH_SIZE, Net, n_channels, n_classes

# Function to visualize image, prediction, and mask

def generate_html(image_paths, save_dir):
    # Sort images by prediction type
    dv_images = []
    cv_images = []
    dvh_images = []

    for image_path in image_paths:
        if 'DV' in image_path:
            dv_images.append(image_path)
        elif 'CV' in image_path:
            cv_images.append(image_path)
        else:
            dvh_images.append(image_path)

    # Generate HTML content
    html_content = '<html><body>\n'

    # Add sections for each prediction type
    if dv_images:
        html_content += '<h2>DV Predictions</h2>\n'
        for image_path in dv_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="DV Prediction Image" style="width:100%;">\n'

    if cv_images:
        html_content += '<h2>CV Predictions</h2>\n'
        for image_path in cv_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="CV Prediction Image" style="width:100%;">\n'

    if dvh_images:
        html_content += '<h2>DVH Predictions</h2>\n'
        for image_path in dvh_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="DVH Prediction Image" style="width:100%;">\n'

    html_content += '</body></html>'

    html_path = os.path.join(save_dir, 'index.html')
    with open(html_path, 'w') as html_file:
        html_file.write(html_content)
    print(f"Saved HTML file to {html_path}")


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

def convert_to_one_hot(prediction, shape, num_classes):
    one_hot_mask = np.zeros((num_classes, *shape), dtype=np.int32)
    for i in range(num_classes):
        one_hot_mask[i, ...] = (prediction == i)
    return one_hot_mask

def create_test_data_loaders(test_image_file, test_mask_file, test_label_file, batch_size):
    images = load_hdf5(test_image_file)['main']
    masks = load_hdf5(test_mask_file)['main']
    labels = load_hdf5(test_label_file)['main']

    dataset = CustomDataset(images=images, labels=labels, masks=masks, transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader, labels


# Function to perform predictions on test images
def test_model(model, data_loader, true_labels, save_dir):
    model.eval()
    y_true = []
    y_pred = []
    num_images_saved = 0  # Counter to keep track of saved images
    image_paths = []  # List to store paths of saved images

    with torch.no_grad():
        for index, (inputs, labels, masks) in enumerate(data_loader):  # Adjusted to unpack three values
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                if num_images_saved >= 5:
                    generate_html(image_paths, save_dir)  # Generate HTML after saving 5 images
                    return

                image = inputs[i, 0].cpu().numpy()
                prediction = predicted[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                pred_label = predicted[i].cpu().item()
                true_label = true_labels[index * inputs.size(0) + i] if index * inputs.size(0) + i < len(true_labels) else 'N/A'

                # Convert the predicted labels to one-hot encoded mask
                prediction_one_hot = convert_to_one_hot(prediction, image.shape, n_classes)

                # Collecting true and predicted labels for evaluation
                y_true.extend(mask.flatten())
                y_pred.extend(prediction.flatten())

                # Visualize the prediction and save the image
                save_path = os.path.join(save_dir, f'prediction_{index * len(inputs) + i + 1}.png')
                visualize_image_with_prediction(image, prediction_one_hot[1], mask, true_label, pred_label, save_dir, index * len(inputs) + i + 1)
                image_paths.append(save_path)
                num_images_saved += 1  # Increment the counter

    # In case fewer than 5 images are processed
    generate_html(image_paths, save_dir)
