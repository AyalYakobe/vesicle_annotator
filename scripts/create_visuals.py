import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import pyvista as pv

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# Function to outline the vesicle
def outline_vesicle(image, threshold=95):
    threshold_value = np.percentile(image, threshold)
    outlined_image = np.zeros_like(image)
    outlined_image[image > threshold_value] = 1  # Use binary mask for highlighting
    return outlined_image

# Function to visualize the final product in 3D
def visualize_final_product_3d(images, prediction, save_path):
    os.makedirs(save_path, exist_ok=True)

    images = images.squeeze(1).cpu().numpy() if isinstance(images, torch.Tensor) else images

    # Sum slices into a 3D image
    combined_image = np.sum(images, axis=0)

    # Normalize the combined image to the range [0, 1]
    combined_image = (combined_image - np.min(combined_image)) / (np.max(combined_image) - np.min(combined_image))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    z, y, x = np.indices(combined_image.shape)
    ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=combined_image.flatten(), cmap='gray', marker='o', s=10, alpha=.9, label='Background')

    if prediction == 1:
        outlined_image = outline_vesicle(combined_image)
        z, y, x = np.indices(outlined_image.shape)
        ax.scatter(x[outlined_image == 1].flatten(), y[outlined_image == 1].flatten(), z[outlined_image == 1].flatten(), c='red', marker='o', s=40, alpha=1.0, label='Vesicle')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Vesicle Present' if prediction == 1 else 'No Vesicle')
    ax.legend()

    plt.savefig(os.path.join(save_path, 'prediction_3d.png'))
    plt.close()

# Function to evaluate and visualize the model
def evaluate_and_visualize_model_3d(model, data_loader, save_path, device='cpu'):
    print("EVALUATING THE MODEL")
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            inputs = inputs.float().unsqueeze(1).to(device)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            if save_path:
                for i in range(inputs.size(0)):
                    visualize_final_product_3d(inputs[i], predicted[i].item(),
                                               os.path.join(save_path, f'batch_{batch_idx}_image_{i}'))



##########PYVISTA#######################PYVISTA#######################PYVISTA#######################PYVISTA#######################PYVISTA#######################PYVISTA#############
##########PYVISTA#######################PYVISTA#######################PYVISTA#######################PYVISTA#######################PYVISTA#######################PYVISTA#############
# Function to convert combined image to a point cloud
# Function to outline the vesicle


# Function to convert combined image to a point cloud
def combined_image_to_point_cloud(combined_image):
    z, y, x = np.indices(combined_image.shape)
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    intensities = combined_image.flatten()
    return points, intensities

# Function to create and save a PyVista point cloud visualization
def visualize_point_cloud(points, intensities, vesicle_mask, save_path):
    point_cloud = pv.PolyData(points)
    point_cloud['intensity'] = intensities

    # Define colors based on intensity values and vesicle mask
    colors = np.zeros((len(intensities), 3))
    # Map intensities to grayscale for background
    normalized_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
    colors[:, 0] = 255 * normalized_intensities  # Red channel
    colors[:, 1] = 255 * normalized_intensities  # Green channel
    colors[:, 2] = 255 * normalized_intensities  # Blue channel
    # Highlight vesicle regions in red
    colors[vesicle_mask == 1, 0] = 255
    colors[vesicle_mask == 1, 1] = 0
    colors[vesicle_mask == 1, 2] = 0

    plotter = pv.Plotter()
    plotter.add_points(point_cloud, scalars=colors / 255.0, rgb=True)
    plotter.add_text('Vesicle Present', position='upper_edge', font_size=10, color='white')
    plotter.show(screenshot=os.path.join(save_path, 'point_cloud.png'))
    plotter.close()

# Function to visualize the final product in 3D using PyVista
def visualize_final_product_pyvista(images, prediction, save_path):
    os.makedirs(save_path, exist_ok=True)

    images = images.squeeze(1).cpu().numpy() if isinstance(images, torch.Tensor) else images

    # Sum slices into a 3D image
    combined_image = np.sum(images, axis=0)

    # Normalize the combined image to the range [0, 1]
    combined_image = (combined_image - np.min(combined_image)) / (np.max(combined_image) - np.min(combined_image))

    points, intensities = combined_image_to_point_cloud(combined_image)
    vesicle_mask = outline_vesicle(combined_image)

    visualize_point_cloud(points, intensities, vesicle_mask.flatten(), save_path)

# Function to evaluate and visualize the model using PyVista
def evaluate_and_visualize_model_pyvista(model, data_loader, save_path, device='cpu'):
    print("EVALUATING THE MODEL")
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            inputs = inputs.float().unsqueeze(1).to(device)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            if save_path:
                for i in range(inputs.size(0)):
                    visualize_final_product_pyvista(inputs[i], predicted[i].item(),
                                                    os.path.join(save_path, f'batch_{batch_idx}_image_{i}'))
