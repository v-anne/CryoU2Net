import os
import random

# Base path to your dataset directory
base_directory = 'new_train_dataset/val'

# Define the decimation factor (e.g., keep 1 file out of every 10)
decimation_factor = 14

def decimate_dataset(directory):
    # Paths to images and masks within the directory
    images_dir = os.path.join(directory, 'images')
    masks_dir = os.path.join(directory, 'masks')

    # List all image files
    image_files = sorted(os.listdir(images_dir))

    # Calculate number of files to keep based on the decimation factor
    num_to_keep = len(image_files) // decimation_factor

    # Randomly select indices of images to keep
    selected_indices = random.sample(range(len(image_files)), num_to_keep)
    images_to_keep = {image_files[i] for i in selected_indices}
    
    # Determine the corresponding masks to keep based on the kept images
    masks_to_keep = {image.replace('.png', '_mask.png') for image in images_to_keep}

    # Delete images that are not kept
    for image_file in image_files:
        if image_file not in images_to_keep:
            os.remove(os.path.join(images_dir, image_file))
            print(f"Deleted {image_file}")
    
    # Delete masks that are not kept
    mask_files = os.listdir(masks_dir)
    for mask_file in mask_files:
        if mask_file not in masks_to_keep:
            os.remove(os.path.join(masks_dir, mask_file))
            print(f"Deleted {mask_file}")

# Apply decimation to both training and validation datasets
decimate_dataset(base_directory)

print("Dataset has been decimated.")