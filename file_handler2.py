import nibabel as nib
import os
import numpy as np

# Function to load a NIfTI file and return its data, affine, and important metadata
def load_nifti_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        nifti_image = nib.load(file_path)
        data = nifti_image.get_fdata()
        affine = nifti_image.affine
        header = nifti_image.header

        # Extract only the important metadata fields
        important_metadata = {
            'datatype': header.get_data_dtype(),
            'dim': header['dim'].copy(),
            'pixdim': header['pixdim'].copy(),
            'qform_code': header['qform_code'],
            'sform_code': header['sform_code'],
            'srow_x': header['srow_x'].copy(),
            'srow_y': header['srow_y'].copy(),
            'srow_z': header['srow_z'].copy(),
            'xyzt_units': header['xyzt_units']
        }

        return data, affine, important_metadata
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"Invalid NIfTI file format: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file {file_path}: {e}")

# Function to create a new NIfTI header with important metadata
def create_header_with_important_metadata(important_metadata):
    new_header = nib.Nifti1Header()
    
    # Set the important metadata fields to the new header
    new_header.set_data_dtype(important_metadata['datatype'])
    new_header['dim'] = important_metadata['dim']
    new_header['pixdim'] = important_metadata['pixdim']
    new_header['qform_code'] = important_metadata['qform_code']
    new_header['sform_code'] = important_metadata['sform_code']
    new_header['srow_x'] = important_metadata['srow_x']
    new_header['srow_y'] = important_metadata['srow_y']
    new_header['srow_z'] = important_metadata['srow_z']
    new_header['xyzt_units'] = important_metadata['xyzt_units']

    return new_header

# Function to save augmented volumes and labels to NIfTI files with important metadata
def save_augmented_volumes(volume_aug, label_aug, output_dir, prefix, affine, important_metadata):
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
            
            # Create a new header with only the important metadata
            new_header = create_header_with_important_metadata(important_metadata)

            # Create NIfTI objects for the augmented volume and label using the new header and original affine
            vol_nifti = nib.Nifti1Image(volume_aug[i], affine=affine, header=new_header)
            lbl_nifti = nib.Nifti1Image(label_aug[i], affine=affine, header=new_header)

            # Construct the output paths for the augmented volume and label
            vol_output_path = os.path.join(output_dir, f'{prefix}_augmented_volume_{i}.nii.gz')
            lbl_output_path = os.path.join(output_dir, f'{prefix}_augmented_label_{i}.nii.gz')

            # Save the augmented volume and label to the specified paths
            nib.save(vol_nifti, vol_output_path)
            nib.save(lbl_nifti, lbl_output_path)
        except Exception as e:
            print(f"Error saving augmented volume or label {i} for prefix {prefix}: {e}")