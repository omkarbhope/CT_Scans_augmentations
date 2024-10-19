import os
from file_handler2 import load_nifti_file, save_augmented_volumes
from augmentation_pipeline import process_volume_and_label
import numpy as np

# Define directories
data_dir = '/Users/omkarbhope/Library/Mobile Documents/com~apple~CloudDocs/Research/PKG - CT-ORG/CT-ORG/TestData'
output_dir = 'augmented_nifti_volumes3'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the list of all volume and label files in the directory, sorted for consistency
volume_files = sorted([f for f in os.listdir(data_dir) if f.startswith('volume') and f.endswith('.nii.gz')])
label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('labels') and f.endswith('.nii.gz')])

# Validate that there is a matching label for each volume
if len(volume_files) != len(label_files):
    raise ValueError("The number of volume files and label files must be the same.")

# Loop through all volume and label files
for volume_file, label_file in zip(volume_files, label_files):
    volume_path = os.path.join(data_dir, volume_file)
    label_path = os.path.join(data_dir, label_file)

    try:
        volume_data, volume_affine, volume_metadata = load_nifti_file(volume_path)
        label_data, label_affine, label_metadata = load_nifti_file(label_path)
        label_data = label_data.astype(np.int32)
    except FileNotFoundError:
        print(f"Error: File not found - {volume_path} or {label_path}")
        continue
    except Exception as e:
        print(f"Error loading files: {e}")
        continue

    # Validate that the volume and label data have the same shape
    if volume_data.shape != label_data.shape:
        print(f"Error: Shape mismatch between volume and label for {volume_file} and {label_file}")
        continue

    # Validate that volume and label data are not empty
    if volume_data.size == 0 or label_data.size == 0:
        print(f"Error: Volume or label data is empty for {volume_file} and {label_file}")
        continue

    # Modify the label data to retain only the values of 1, putting 0 everywhere else
    label_data_modified = np.where(label_data == 3, 1, 0)

    # Convert the modified label data to float64 for consistency with the processing pipeline
    label_data_modified = label_data_modified.astype(np.float64)

    # Generate augmented volumes and labels using the process_volume_and_label function
    try:
        volume_aug, label_aug = process_volume_and_label(volume_data, label_data_modified)
    except Exception as e:
        print(f"Error during augmentation processing for {volume_file}: {e}")
        continue

    # Validate that augmentations were generated correctly
    if len(volume_aug) != len(label_aug):
        print(f"Error: Mismatch between augmented volumes and labels for {volume_file}")
        continue

    # Save the augmented volumes and labels using the save_augmented_volumes function
    try:
        save_augmented_volumes(volume_aug, label_aug, output_dir, prefix=volume_file.split('.')[0], affine=volume_affine, important_metadata=volume_metadata)
    except Exception as e:
        print(f"Error saving augmented volumes for {volume_file}: {e}")
        continue

print("Processing completed.")
