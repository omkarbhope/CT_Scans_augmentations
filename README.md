# Project Overview

This project is focused on processing, augmenting, and handling NIfTI medical imaging files. It includes various utilities to manage medical data, perform augmentations, and split datasets for machine learning models.


## Dataset

To download the dataset you can go to: "https://www.cancerimagingarchive.net/collection/ct-org/"
You will also need IBM Aspera Connect 

## Directory Structure

The `Code` directory contains the following scripts:

- **`file_handler2.py`**: Handles file I/O operations such as loading and saving NIfTI files.
- **`augmentation_pipeline.py`**: Implements the data augmentation pipeline for medical imaging data.
- **`process_nifty2.py`**: Processes NIfTI files for data preparation and augmentation.
- **`main2.py`**: The main script to run the application, combining data handling, augmentation, and processing steps.
- **`split_data.py`**: Utility script to split the dataset into training, validation, and testing sets.
- **`augmentation_utils.py`**: Contains utility functions used for image augmentation.

## Requirements

This project requires the following packages:

- `numpy`
- `scikit-learn`
- `nibabel` (for handling NIfTI files)
- `opencv-python` (for image processing)
- `matplotlib` (for visualization)

You can install the dependencies using the following command:

```sh
pip install numpy scikit-learn nibabel opencv-python matplotlib
```

## Running the Code

1. **Prepare the Augmented Files**: Ensure that your NIfTI volume and label files are prepared and available.

2. **Run the Scripts**:

   - **Augmentation**: Use `augmentation_pipeline.py` to perform data augmentation on the NIfTI files.
   - **Dataset Splitting**: Run `split_data.py` to split the dataset into training, validation, and testing sets.
   - **Main Application**: Execute `main2.py` to process the data and apply all augmentations in sequence.

```sh
python main2.py
```

3. **Output**: The processed and augmented files will be saved in the specified output directories, and the dataset will be organized into train, validation, and test splits.

## File Descriptions

- **`file_handler2.py`**: This script provides functionality to load and save NIfTI files, which is crucial for managing medical imaging data.
- **`augmentation_pipeline.py`**: This script defines functions that apply various augmentations, such as rotation, flipping, zooming, and contrast adjustment to medical images.
- **`process_nifty2.py`**: This script contains code for preprocessing the individual NIfTI volumes, including resizing, normalization, and label adjustments finally create a visualization window displaying all the augmented slices with their corresponding labels.
- **`split_data.py`**: This script splits the dataset into training, validation, and testing sets in a configurable ratio (default is 70/20/10).
- **`augmentation_utils.py`**: Contains utility functions that assist with augmentations, ensuring reusability and consistency.

## License

This project is licensed under the MIT License.

## Author

Omkar Bhope

