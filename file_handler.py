import nibabel as nib
import os
import numpy as np

# Function to load a NIfTI file and return its data
def load_nifti_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        nifti_image = nib.load(file_path)
        return nifti_image.get_fdata()
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"Invalid NIfTI file format: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file {file_path}: {e}")

# Function to save augmented volumes and labels to NIfTI files
def save_augmented_volumes(volume_aug, label_aug, output_dir, prefix):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Validate that volume_aug and label_aug are lists and have the same length
    if not isinstance(volume_aug, list) or not isinstance(label_aug, list):
        raise TypeError("volume_aug and label_aug must be lists.")
    if len(volume_aug) != len(label_aug):
        raise ValueError("volume_aug and label_aug must have the same length.")

    # Loop through each of the augmented volumes and labels
    for i in range(len(volume_aug)):
        try:
            # Validate that the augmented volumes and labels are not empty
            if volume_aug[i].size == 0 or label_aug[i].size == 0:
                raise ValueError(f"Augmented volume or label {i} is empty.")
            
            # Create NIfTI objects for the augmented volume and label
            vol_nifti = nib.Nifti1Image(volume_aug[i], affine=np.eye(4))
            lbl_nifti = nib.Nifti1Image(label_aug[i], affine=np.eye(4))

            # Construct the output paths for the augmented volume and label
            vol_output_path = os.path.join(output_dir, f'{prefix}_augmented_volume_{i}.nii.gz')
            lbl_output_path = os.path.join(output_dir, f'{prefix}_augmented_label_{i}.nii.gz')

            # Save the augmented volume and label to the specified paths
            nib.save(vol_nifti, vol_output_path)
            nib.save(lbl_nifti, lbl_output_path)
        except Exception as e:
            print(f"Error saving augmented volume or label {i} for prefix {prefix}: {e}")