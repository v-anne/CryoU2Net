import os

# Set the directory paths for images and masks
images_dir = 'train_dataset/val/images'
masks_dir = 'train_dataset/val/masks'

# List all files in the images directory
images = os.listdir(images_dir)

# Iterate over each image file
for image in images:
    # Construct the expected mask filename
    mask_name = image.replace('.jpg', '_mask.jpg')

    # Check if the mask file exists in the masks directory
    if not os.path.exists(os.path.join(masks_dir, mask_name)):
        # If the mask does not exist, delete the image
        os.remove(os.path.join(images_dir, image))
        print(f"Deleted: {image}")
    # else:
    #     print(f"Match found for: {image}")

print("Cleanup complete.")
