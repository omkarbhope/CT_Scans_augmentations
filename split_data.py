import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Assuming 'volumes' and 'labels' are lists containing paths to the augmented volume and label files respectively
data_dir = "augmented_nifti_volumes2"

# Get the list of volume and label files
volume_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'augmented_volume' in f and f.endswith('.nii.gz')])
label_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'augmented_label' in f and f.endswith('.nii.gz')])

# Ensure there is a matching label for each volume
if len(volume_files) != len(label_files):
    raise ValueError("The number of volume files and label files must be the same.")

# Ensure there are enough samples to split
if len(volume_files) < 2:
    raise ValueError("Not enough samples to perform train/validation/test split. Ensure there are at least 2 samples in the dataset.")

# Split the dataset into training (70%), validation (20%), and testing (10%)
try:
    vol_train, vol_temp, lbl_train, lbl_temp = train_test_split(volume_files, label_files, test_size=0.3, random_state=42)
    vol_val, vol_test, lbl_val, lbl_test = train_test_split(vol_temp, lbl_temp, test_size=1/3, random_state=42)
except Exception as e:
    raise RuntimeError(f"Error during dataset splitting: {e}")

# Save the split file paths for later use
splits = {
    'train': (vol_train, lbl_train),
    'val': (vol_val, lbl_val),
    'test': (vol_test, lbl_test)
}

# Create directories for splits and move files
for split_name, (volumes, labels) in splits.items():
    split_dir = f"{data_dir}/{split_name}"
    os.makedirs(split_dir, exist_ok=True)
    for volume_path, label_path in zip(volumes, labels):
        try:
            # Move the files to their respective folders
            shutil.move(volume_path, f"{split_dir}/{os.path.basename(volume_path)}")
            shutil.move(label_path, f"{split_dir}/{os.path.basename(label_path)}")
        except FileNotFoundError:
            print(f"File not found: {volume_path} or {label_path}")
        except Exception as e:
            print(f"Error moving files {volume_path} and {label_path}: {e}")

print("Dataset splitting and file organization completed successfully.")