# Vesicle Detection and Visualization

This project aims to detect vesicles in neuron images using a modified version of the MedMNIST platform, adapted to handle HDF5 files. The detected vesicles are visualized in 3D, highlighting the regions where vesicles are present.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Running the Code](#running-the-code)
- [Output](#output)
- [Handling HDF5 Files](#handling-hdf5-files)

## Overview

The primary goal of this project is to identify vesicles within neuron images stored in HDF5 files and visualize these detections in a 3D space. The project involves the following main steps:
1. **Data Loading**: Read image, label, and mask data from HDF5 files.
2. **Model Training**: Train a 3D convolutional neural network to detect vesicles.
3. **Evaluation and Visualization**: Evaluate the model on the test dataset and visualize the vesicle detections in 3D.

## Setup

To set up the project, follow these steps:

1. **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd vesicle_detection
    ```

2. **Install Dependencies**
    Install the necessary Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare HDF5 Files**
    Ensure that you have the following HDF5 files in the specified directory:
    - `bigV_cls_im.h5`: Contains image data.
    - `bigV_cls_label.h5`: Contains label data.
    - `bigV_cls_mask.h5`: Contains mask data.

    Place these files in the `/Users/ayalyakobe/vesicle_annotations` directory or update the paths in the code accordingly.

## Running the Code

1. **Train the Model**
    Run the main script to train the model and evaluate it:
    ```bash
    python main.py
    ```

    The script performs the following tasks:
    - Loads the HDF5 files and preprocesses the data.
    - Creates a 3D convolutional neural network model.
    - Trains the model on the training data.
    - Evaluates the model on the test data.
    - Visualizes the model's predictions in 3D.

## Output

The output includes:
- **Training Logs**: Printed to the console, showing the progress of the training process.
- **Model Checkpoints**: Saved during training to `model_checkpoint.pth`.
- **Evaluation Metrics**: Printed to the console after evaluation, including accuracy, precision, recall, and F1 score.
- **3D Visualizations**: Saved as images in the specified output directory, showing the detected vesicles.

## Handling HDF5 Files

To adapt the MedMNIST imaging platform for our specific task of identifying vesicles, significant modifications were necessary to handle HDF5 files, which contain our image, label, and mask data. The original MedMNIST codebase was designed for simpler image file formats, so extensive changes were made to ensure compatibility with HDF5 structures. These modifications included implementing functions for reading and writing HDF5 files and ensuring that the data was correctly loaded into the model for training and evaluation. After extensive debugging and iterative testing, the altered code successfully processed the HDF5 files, marking a critical milestone in adapting the platform for our use case.

---

This README provides a comprehensive overview of the project, including the necessary setup, running instructions, and an explanation of how HDF5 files are handled.
