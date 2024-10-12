import numpy as np
from augmentation_utils import *

# Function to process an individual image and its corresponding label, applying all augmentations
def process_image_and_label(image, label):
    if image is None or label is None or image.size == 0 or label.size == 0:
        raise ValueError("Input image or label is empty or None.")
    if image.shape != label.shape:
        raise ValueError("Image and label must have the same shape.")

    # Convert the image to grayscale and normalize both image and label to 512x512 pixels
    try:
        grayscale_image = convert_to_grayscale(normalize_image(image))
        normalized_label = normalize_image(label)
    except Exception as e:
        raise RuntimeError(f"Error during normalization or grayscale conversion: {e}")

    # Apply various augmentations to the image and label
    try:
        rotated_image, rotated_label = rotate_image_and_label(grayscale_image, normalized_label)
        flipped_image, flipped_label = flip_image_and_label(grayscale_image, normalized_label)
        zoomed_image, zoomed_label = zoom_image_and_label(grayscale_image, normalized_label)
        contrast_image, contrast_label = adjust_contrast(grayscale_image, normalized_label)
        denoised_image, denoised_label = reduce_noise(grayscale_image, normalized_label)
    except Exception as e:
        raise RuntimeError(f"Error during augmentation: {e}")

    # Return all augmented images and labels
    return [grayscale_image, rotated_image, flipped_image, zoomed_image, contrast_image, denoised_image], \
           [normalized_label, rotated_label, flipped_label, zoomed_label, contrast_label, denoised_label]

# Function to process an entire volume and its corresponding label, applying all augmentations slice by slice
def process_volume_and_label(volume, label):
    if volume is None or label is None or volume.size == 0 or label.size == 0:
        raise ValueError("Input volume or label is empty or None.")
    if volume.shape != label.shape:
        raise ValueError("Volume and label must have the same shape.")

    # Create lists to store the augmented volumes and labels for all slices
    augmentations_volume = [[] for _ in range(6)]
    augmentations_label = [[] for _ in range(6)]

    # Loop through each slice in the volume and label
    for i in range(volume.shape[2]):
        image = volume[:, :, i]
        label_slice = label[:, :, i]

        # Apply augmentations to each slice
        try:
            augmentations, label_augmentations = process_image_and_label(image, label_slice)
        except Exception as e:
            print(f"Error processing slice {i}: {e}")
            continue

        # Store each augmented slice in the respective list
        for j in range(6):
            augmentations_volume[j].append(augmentations[j])
            augmentations_label[j].append(label_augmentations[j])

    # Validate that we have successfully processed slices
    if any(len(aug) == 0 for aug in augmentations_volume):
        raise RuntimeError("Failed to process any slices for augmentation.")

    # Stack the augmented slices along the third dimension to form complete volumes
    try:
        augmentations_volume = [np.stack(aug, axis=2) for aug in augmentations_volume]
        augmentations_label = [np.stack(aug, axis=2) for aug in augmentations_label]
    except Exception as e:
        raise RuntimeError(f"Error stacking augmented slices: {e}")

    # Return the augmented volumes and labels
    return augmentations_volume, augmentations_label