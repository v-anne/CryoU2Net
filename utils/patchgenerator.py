import cv2
import os
import numpy as np

def create_patches(img_dir, mask_dir, output_img_dir, output_mask_dir, patch_size=1024):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    
    img_files = os.listdir(img_dir)
    for img_filename in img_files:
        img_path = os.path.join(img_dir, img_filename)
        mask_path = os.path.join(mask_dir, img_filename.replace('.jpg', '_mask.jpg'))
        
        if not os.path.exists(mask_path):
            print(f"No mask file for {img_filename}. Skipping this image.")
            continue
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error loading image or mask for {img_filename}. Check file paths and integrity.")
            continue

        if image.shape[0] < patch_size or image.shape[1] < patch_size or mask.shape[0] < patch_size or mask.shape[1] < patch_size:
            print(f"Image or mask dimensions are smaller than the patch size for {img_filename}. Skipping this file.")
            continue

        for i in range(0, image.shape[0] - patch_size + 1, patch_size):
            for j in range(0, image.shape[1] - patch_size + 1, patch_size):
                img_patch = image[i:i + patch_size, j:j + patch_size]
                mask_patch = mask[i:i + patch_size, j:j + patch_size]
                
                patch_filename = f'{img_filename[:-4]}_{i}_{j}.png'
                cv2.imwrite(os.path.join(output_img_dir, patch_filename), img_patch)
                cv2.imwrite(os.path.join(output_mask_dir, patch_filename.replace('.png', '_mask.png')), mask_patch)
        print(img_filename)

def process_datasets(base_dir, new_base_dir, patch_size=1024):
    train_img_dir = os.path.join(base_dir, 'images')
    train_mask_dir = os.path.join(base_dir, 'masks')
    new_train_img_dir = os.path.join(new_base_dir, 'images')
    new_train_mask_dir = os.path.join(new_base_dir, 'masks')
    create_patches(train_img_dir, train_mask_dir, new_train_img_dir, new_train_mask_dir, patch_size)

    val_img_dir = os.path.join(base_dir, 'val/images')
    val_mask_dir = os.path.join(base_dir, 'val/masks')
    new_val_img_dir = os.path.join(new_base_dir, 'val/images')
    new_val_mask_dir = os.path.join(new_base_dir, 'val/masks')
    create_patches(val_img_dir, val_mask_dir, new_val_img_dir, new_val_mask_dir, patch_size)

# Example usage:
process_datasets('decimation_dataset', 'new_train_dataset')
