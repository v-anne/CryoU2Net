# Code for creating dataset

from torch.utils.data import Dataset
import mrcfile
import cv2
import torch
import config

def min_max(image):
    i_min = image.min()
    i_max = image.max()

    image = ((image - i_min)/(i_max - i_min))
    return image

def transform(image):
    i_min = image.min()
    i_max = image.max()
    
    if i_max == 0:
        return image

    image = ((image - i_min)/(i_max - i_min)) * 255
    return image.astype('uint8')


# class CryoEMDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, transform=None, patch_size=1024):
#         super().__init__()
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.patch_size = patch_size

#     def __len__(self):
#         total_patches = 0
#         for img_path in self.img_dir:
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             h, w = img.shape
#             total_patches += (h // self.patch_size) * (w // self.patch_size)
#         return total_patches

#     def __getitem__(self, idx):
#         global_patch_index = 0
#         total_images = len(self.img_dir)
#         for image_index, (img_path, mask_path) in enumerate(zip(self.img_dir, self.mask_dir)):
#             image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             num_patches_per_row = image.shape[1] // self.patch_size
#             num_patches_per_image = (image.shape[0] // self.patch_size) * num_patches_per_row

#             if idx < global_patch_index + num_patches_per_image:
#                 local_patch_index = idx - global_patch_index
#                 row_index = local_patch_index // num_patches_per_row
#                 col_index = local_patch_index % num_patches_per_row
#                 top = row_index * self.patch_size
#                 left = col_index * self.patch_size
#                 image_patch = image[top:top + self.patch_size, left:left + self.patch_size]
#                 mask_patch = mask[top:top + self.patch_size, left:left + self.patch_size]
                
#                 if self.transform:
#                     image_patch = self.transform(image_patch)
#                     mask_patch = self.transform(mask_patch)

#                 image_patch = torch.from_numpy(image_patch).unsqueeze(0).float() / 255.0
#                 mask_patch = torch.from_numpy(mask_patch).unsqueeze(0).float() / 255.0
                
#                 # Progress print statement
#                 print(f"Processing image {image_index+1}/{total_images}, patch {local_patch_index+1}/{num_patches_per_image}")
                
#                 return (image_patch, mask_patch)

#             global_patch_index += num_patches_per_image

#         raise IndexError("Index out of range in dataset")

class CryoEMDataset(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        # image = mrcfile.read(self.img_dir[idx])
        # image = image.T
        # image = np.rot90(image)
        
        image_path = self.img_dir[idx]
        mask_path = image_path[:-4] + '_mask.jpg' # png or jpg?
        mask_path = mask_path.replace('images', 'masks')
        
        image = cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path, 0)
        
        image = cv2.resize(image, (config.input_image_width, config.input_image_height))
        mask = cv2.resize(mask, (config.input_image_width, config.input_image_height))
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        image = image/255.0
        mask = mask/255.0

        return (image, mask)
    
class CryoEMFineTuneDataset(Dataset):
    def __init__(self, mask_dir, transform):
        super().__init__()
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.mask_dir)

    def __getitem__(self, idx):        
        mask_path = self.mask_dir[idx]
        image_path = mask_path[:-9] + '.jpg'
        image_path = image_path.replace('masks', 'images')
        #image = denoise(image_path)
        image = cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path, 0)
        
        image = cv2.resize(image, (config.input_image_width, config.input_image_height))
        mask = cv2.resize(mask, (config.input_image_width, config.input_image_height))
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        image = image/255.0
        mask = mask/255.0

        return (image, mask)