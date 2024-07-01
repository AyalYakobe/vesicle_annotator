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
