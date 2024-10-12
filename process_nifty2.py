import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
from augmentation_utils import *
from augmentation_pipeline import process_image_and_label

# Load the CT volume and label data using nibabel
volume_path = '/Users/omkarbhope/Library/Mobile Documents/com~apple~CloudDocs/Research/PKG - CT-ORG/CT-ORG/volume-0.nii.gz'  # Replace with the actual path to your volume file
label_path = '/Users/omkarbhope/Library/Mobile Documents/com~apple~CloudDocs/Research/PKG - CT-ORG/CT-ORG/labels-0.nii.gz'  # Replace with the actual path to your label file

try:
    volume = nib.load(volume_path)
    labels = nib.load(label_path)
except FileNotFoundError:
    raise FileNotFoundError("Volume or label file not found. Please provide the correct file paths.")
except nib.filebasedimages.ImageFileError:
    raise ValueError("The provided file is not a valid NIFTI file. Please check the file format.")

volume_data = volume.get_fdata()
label_data = labels.get_fdata().astype(np.int32)  # Convert label data to int

# Validate data dimensions
if volume_data.shape != label_data.shape:
    raise ValueError("Volume data and label data must have the same shape.")

# Validate that volume and label data are not empty
if volume_data.size == 0 or label_data.size == 0:
    raise ValueError("Volume data or label data is empty. Please provide valid data.")

# Modify label data to retain values of 1 and put 0 everywhere else
label_data_modified = np.where(label_data == 3, 1, 0)

# Initial slice index
current_slice = label_data.shape[2] // 2

# Create a function to plot the slice and augmentations
def plot_slice(slice_index):
    plt.suptitle('CT Scan Viewer', fontsize=16)
    for ax in axs:
        ax.clear()
    
    # Display the CT volume slice and corresponding label slice
    ct_slice = volume_data[:, :, slice_index]
    label_slice = label_data_modified[:, :, slice_index]
    
    # Validate that the slice is not empty
    if np.sum(ct_slice) == 0:
        print(f"Warning: Slice {slice_index} is empty.")
    
    try:
        augmentations, label_augmentations = process_image_and_label(ct_slice, label_slice)
    except Exception as e:
        raise RuntimeError(f"Error during augmentation processing: {e}")
    
    # Validate that augmentations match expected output
    if len(augmentations) != len(label_augmentations):
        raise ValueError("Mismatch between augmentations and label augmentations.")
    
    titles = ["Original Grayscale", "Rotated", "Flipped", "Zoomed", "Contrast Adjusted", "Denoised"]
    
    for i, (aug_image, aug_label, title) in enumerate(zip(augmentations, label_augmentations, titles)):
        axs[i].imshow(aug_image, cmap='gray')
        axs[i].set_title(f'{title} - Image')
        axs[i].axis('off')
        # Plot corresponding label in the next available axis
        axs[i + 6].imshow(aug_label, cmap='viridis')
        axs[i + 6].set_title(f'{title} - Label')
        axs[i + 6].axis('off')

    plt.pause(0.01)
    plt.draw()

# Create the figure and initial plot
fig, axs = plt.subplots(2, 6, figsize=(24, 10))
axs = axs.ravel()[:12]  # Use only 12 axes to plot augmentations and labels
plt.subplots_adjust(bottom=0.2)
plot_slice(current_slice)

# Add buttons for navigating slices
ax_prev = plt.axes([0.4, 0.01, 0.1, 0.05])
ax_next = plt.axes([0.51, 0.01, 0.1, 0.05])

prev_button = Button(ax_prev, 'Previous')
next_button = Button(ax_next, 'Next')

# Update functions for buttons
def prev_slice(event):
    global current_slice
    if current_slice > 0:
        current_slice -= 1
        plot_slice(current_slice)
    else:
        print("Already at the first slice.")

def next_slice(event):
    global current_slice
    if current_slice < label_data.shape[2] - 1:
        current_slice += 1
        plot_slice(current_slice)
    else:
        print("Already at the last slice.")

prev_button.on_clicked(prev_slice)
next_button.on_clicked(next_slice)

plt.show()