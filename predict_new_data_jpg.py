# Code for making predictions on individual micrographs

import copy
from utils.denoise import denoise, denoise_jpg_image
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import glob
import os
from dataset.dataset import transform
from models.model_5_layers import UNET
from models.u2net import U2NETP as U2NET
import config
import mrcfile
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import statistics as st

print("[INFO] Loading up model...")
model = U2NET().to(device=config.device)
# model2 = UNET().to(device=config.device)
state_dict = torch.load("weights/CryoSegNet.pth") #???
# state_dict2 = torch.load("weights/cryosegnet.pth")
model.load_state_dict(state_dict)
# model2.load_state_dict(state_dict2)

sam_model = sam_model_registry[config.model_type](checkpoint=config.sam_checkpoint)
sam_model.to(device=config.device)

mask_generator = SamAutomaticMaskGenerator(sam_model)

def get_annotations(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    return img

def prepare_plot(image, predicted_mask, sam_mask, coords, image_path):
    plt.figure(figsize=(40, 30))
    plt.subplot(221)
    plt.title('Image', fontsize=14)
    plt.imshow(image, cmap='gray')
    plt.subplot(222)
    plt.title('U-NET Mask', fontsize=14)
    plt.imshow(predicted_mask, cmap='gray')
    plt.subplot(223)
    plt.title('SAM Mask', fontsize=14)
    plt.imshow(sam_mask, cmap='gray')
    plt.subplot(224)
    plt.title('Final Picked Particles', fontsize=14)
    plt.imshow(coords, cmap='gray')
    path = image_path.split("/")[-1]
    path = path.replace(".png", "_result.png")
    path = path.replace(".jpg", "_result.jpg")
    final_path = os.path.join(f"{config.output_path}/results/", f'{path}')
    cv2.imwrite(final_path, coords)
    final_path = final_path.replace("result.png", "composite.png")
    final_path = final_path.replace("result.jpg", "composite.jpg")
    plt.savefig(f"{final_path}")
    final_path = final_path.replace("composite.png", "predicted_mask.png")
    final_path = final_path.replace("composite.jpg", "predicted_mask.jpg")
    plt.imsave(final_path, predicted_mask, cmap='gray')
    final_path = final_path.replace("predicted_mask.png", "sam_mask.png")
    final_path = final_path.replace("predicted_mask.jpg", "sam_mask.jpg")
    plt.imsave(final_path, sam_mask, cmap='gray')
    plt.close()



def make_predictions(model, image_path):
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        image = cv2.imread(image_path, 0)
        
        #Check if denoising makes difference or not! If the images are already denoised don't denoise them else denoise them!
        # image = denoise_jpg_image(image)
        print(image)
        height, width = image.shape
        orig_image = copy.deepcopy(image)
        image = cv2.resize(image, (config.input_image_width, config.input_image_height))    
        segment_mask = copy.deepcopy(orig_image)
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image / 255.0
        image = image.to(config.device).unsqueeze(0)
        
        
        predicted_mask = model(image)[0]  
        # print(predicted_mask)
     
        predicted_mask = torch.sigmoid(predicted_mask)
        predicted_mask = predicted_mask.cpu().numpy().reshape(config.input_image_width, config.input_image_height)
        
        # Lower the threshold for detection
        threshold = 0.3  # Adjust this value based on your needs
        predicted_mask = (predicted_mask > threshold).astype(np.float32)

        sam_output = np.repeat(transform(predicted_mask)[:,:,None], 3, axis=-1)
        predicted_mask = cv2.resize(predicted_mask, (width, height))
        
        masks = mask_generator.generate(sam_output)
        sam_mask = get_annotations(masks)
        sam_mask = cv2.resize(sam_mask, (width, height) )
        
        
        
        bboxes = []
        for i in range(0, len(masks)):
            if masks[i]["predicted_iou"] > 0.40:
                box = masks[i]["bbox"]
                bboxes.append(box)
        
        if len(bboxes) > 1:
        
            x_ = st.mode([box[2] for box in bboxes])
            y_ = st.mode([box[3] for box in bboxes])
            d_ = np.sqrt((x_ * width / config.input_image_width)**2 + (y_ * height / config.input_image_height)**2)
            scale_factor = 0.75
            r_ = int((d_ // 2) * scale_factor)
            th = r_ * 0.25
            segment_mask = cv2.cvtColor(segment_mask, cv2.COLOR_GRAY2BGR)
            for b in bboxes:
                if b[2] < x_ + th and b[2] > x_ - th/3 and b[3] < y_ + th and b[3] > y_ - th/3: 
                    x_new, y_new = int((b[0] + b[2]/2) / config.input_image_width * width) , int((b[1] + b[3]/2) / config.input_image_height * height)
                    coords = cv2.circle(segment_mask, (x_new, y_new),  r_, (0, 0, 255), 8)
            try:
                prepare_plot(orig_image, predicted_mask, sam_mask, coords, image_path)
            except:
                pass
        else:
            pass
        
print("[INFO] Loading up test images path ...")
images_path = list(glob.glob(f"output/patches/*.*p*g"))
print(f"{len(images_path)} images.")

for i in range(0, len(images_path), 1):
    make_predictions(model, images_path[i])
    print(i)