import cv2
import numpy as np

# Function to normalize the input image to a size of 512x512 pixels
def normalize_image(image):
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    return cv2.resize(image, (512, 512))

# Function to convert an image to grayscale
# If the input image has three channels, it is converted to a single channel grayscale image
def convert_to_grayscale(image):
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        return image
    else:
        raise ValueError("Input image has an unsupported number of channels.")

# Function to rotate both an image and its corresponding label by a specified angle
def rotate_image_and_label(image, label, angle=30):
    if image is None or label is None or image.size == 0 or label.size == 0:
        raise ValueError("Input image or label is empty or None.")
    if image.shape != label.shape:
        raise ValueError("Image and label must have the same shape.")
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    rotated_label = cv2.warpAffine(label, rotation_matrix, (cols, rows), flags=cv2.INTER_NEAREST)
    return rotated_image, rotated_label

# Function to flip both an image and its corresponding label horizontally
def flip_image_and_label(image, label):
    if image is None or label is None or image.size == 0 or label.size == 0:
        raise ValueError("Input image or label is empty or None.")
    if image.shape != label.shape:
        raise ValueError("Image and label must have the same shape.")
    flipped_image = cv2.flip(image, 1)
    flipped_label = cv2.flip(label, 1)
    return flipped_image, flipped_label

# Function to zoom into both an image and its corresponding label
# The central region of the image is cropped and then resized back to 512x512 pixels
def zoom_image_and_label(image, label):
    if image is None or label is None or image.size == 0 or label.size == 0:
        raise ValueError("Input image or label is empty or None.")
    if image.shape != label.shape:
        raise ValueError("Image and label must have the same shape.")
    if image.shape[0] < 462 or image.shape[1] < 462:
        raise ValueError("Input image and label must be at least 462x462 pixels for zooming.")
    zoomed_image = image[50:462, 50:462]
    zoomed_label = label[50:462, 50:462]
    return cv2.resize(zoomed_image, (512, 512)), cv2.resize(zoomed_label, (512, 512), interpolation=cv2.INTER_NEAREST)

# Function to adjust the contrast of the image, with no changes to the label
def adjust_contrast(image, label, alpha=0.35, beta=0):
    if image is None or label is None or image.size == 0 or label.size == 0:
        raise ValueError("Input image or label is empty or None.")
    contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_image, label

# Function to reduce noise in the image using Gaussian blur, with no changes to the label
def reduce_noise(image, label):
    if image is None or label is None or image.size == 0 or label.size == 0:
        raise ValueError("Input image or label is empty or None.")
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image, label