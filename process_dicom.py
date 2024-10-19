import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime

# Folder containing DICOM files
dicom_folder_path = "/Users/omkarbhope/Library/Mobile Documents/com~apple~CloudDocs/Research/CT_Images/100002/1.2.840.113654.2.55.187766322555605983451267194286230980878/1.2.840.113654.2.55.122344168497038128022524906545138736420"  # Replace with your folder path

# Output folder for augmented DICOM files
output_folder_path = "./augmented_dicoms"
os.makedirs(output_folder_path, exist_ok=True)

# Get list of DICOM files
dicom_files = [f for f in os.listdir(dicom_folder_path) if f.endswith(".dcm")]

# Function for Normalization: Resize the image to 512x512 pixels
def normalize_image(image):
    return cv2.resize(image, (512, 512))

# Function for Color Space Conversion: Convert the image to grayscale
def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Function for Data Augmentation: Apply rotation
def rotate_image(image, angle=30):
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Function for Data Augmentation: Apply flipping
def flip_image(image):
    return cv2.flip(image, 1)  # Horizontal flip

# Function for Data Augmentation: Apply zooming (cropping and resizing back to original size)
def zoom_image(image):
    zoomed_image = image[50:462, 50:462]  # Crop the central region
    return cv2.resize(zoomed_image, (512, 512))

# Function for Data Augmentation: Adjust contrast
def adjust_contrast(image, alpha=0.15, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Function for Noise Reduction: Apply Gaussian blur
def reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Function to save the augmented image as DICOM
def save_augmented_dicom(file_name, original_dicom, augmented_image, suffix):
    # Create a new DICOM dataset
    new_dicom = FileDataset(None, {}, file_meta=original_dicom.file_meta, preamble=b"\0" * 128)
    new_dicom.PixelData = augmented_image.tobytes()
    new_dicom.Rows, new_dicom.Columns = augmented_image.shape
    new_dicom.BitsAllocated = original_dicom.BitsAllocated
    new_dicom.BitsStored = original_dicom.BitsStored
    new_dicom.HighBit = original_dicom.HighBit
    new_dicom.PixelRepresentation = original_dicom.PixelRepresentation
    new_dicom.SamplesPerPixel = original_dicom.SamplesPerPixel
    new_dicom.PhotometricInterpretation = original_dicom.PhotometricInterpretation
    new_dicom.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
    new_dicom.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
    new_dicom.SOPInstanceUID = pydicom.uid.generate_uid()
    
    # Update metadata
    new_dicom.PatientName = original_dicom.PatientName
    new_dicom.PatientID = original_dicom.PatientID
    new_dicom.StudyInstanceUID = original_dicom.StudyInstanceUID
    new_dicom.SeriesInstanceUID = original_dicom.SeriesInstanceUID
    new_dicom.ImagePositionPatient = original_dicom.ImagePositionPatient
    new_dicom.PixelSpacing = original_dicom.PixelSpacing
    
    # Save the new DICOM file
    output_path = os.path.join(output_folder_path, f"{file_name}_{suffix}.dcm")
    new_dicom.save_as(output_path)

# Function to process an image and save all augmentations
def process_image(image_path, file_name):
    dicom_data = pydicom.dcmread(image_path)
    image = dicom_data.pixel_array
    
    # Apply each processing step
    normalized_image = normalize_image(image)
    grayscale_image = convert_to_grayscale(normalized_image)
    rotated_image = rotate_image(grayscale_image)
    flipped_image = flip_image(grayscale_image)
    zoomed_image = zoom_image(grayscale_image)
    contrast_image = adjust_contrast(grayscale_image)
    denoised_image = reduce_noise(grayscale_image)
    
    # Save all augmented images as DICOM
    save_augmented_dicom(file_name, dicom_data, grayscale_image, "grayscale")
    save_augmented_dicom(file_name, dicom_data, rotated_image, "rotated")
    save_augmented_dicom(file_name, dicom_data, flipped_image, "flipped")
    save_augmented_dicom(file_name, dicom_data, zoomed_image, "zoomed")
    save_augmented_dicom(file_name, dicom_data, contrast_image, "contrast")
    save_augmented_dicom(file_name, dicom_data, denoised_image, "denoised")

# Process each DICOM file in the folder
for dicom_file in dicom_files:
    process_image(os.path.join(dicom_folder_path, dicom_file), dicom_file)
