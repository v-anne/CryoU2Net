# Code for training

from utils.accuracy import dice_score, jaccard_score
from dataset.dataset import CryoEMDataset
from models.model_5_layers import UNET
from models.u2net import U2NETP as U2NET
import numpy as np
import config
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loss import DiceLoss
import glob
from tqdm import tqdm
import time
from datetime import datetime, date
import os


# # ------- 1. define loss function --------

# bce_loss = nn.BCELoss(size_average=True)

# def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

#     loss0 = bce_loss(d0,labels_v)
#     loss1 = bce_loss(d1,labels_v)
#     loss2 = bce_loss(d2,labels_v)
#     loss3 = bce_loss(d3,labels_v)
#     loss4 = bce_loss(d4,labels_v)
#     loss5 = bce_loss(d5,labels_v)
#     loss6 = bce_loss(d6,labels_v)

#     loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
#     print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

#     return loss0, loss


# load the image

train_image_path = list(glob.glob(config.train_dataset_path + 'images/*.png'))

val_image_path = list(glob.glob(config.train_dataset_path + 'val/images/*.png'))

print(len(train_image_path))

train_ds = CryoEMDataset(img_dir=train_image_path, transform=None)
val_ds = CryoEMDataset(img_dir=val_image_path, transform=None)

print(f"[INFO] Found {len(train_ds)} examples in the training set...")
print(f"[INFO] Found {len(val_ds)} examples in the validation set...")

train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.batch_size, pin_memory=config.pin_memory, num_workers=config.num_workers)
val_loader = DataLoader(val_ds, shuffle=True, batch_size=config.batch_size, pin_memory=config.pin_memory, num_workers=config.num_workers)
print(f"[INFO] Train Loader Length {len(train_loader)}...")

# initialize our U2-Net model
# model = UNET().to(config.device)
model = U2NET().to(config.device)

# initialize loss function and optimizer
criterion1 = BCEWithLogitsLoss()
criterion2 = DiceLoss()
optimizer = Adam(model.parameters(), lr=config.learning_rate)


# calculate steps per epoch for training and test set
train_steps = len(train_ds) // config.batch_size
val_steps = len(val_ds) // config.batch_size
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"[INFO] Number of Training Steps : {train_steps}")
print(f"[INFO] Number of Validation Steps : {val_steps}")
#print(f"[INFO] Total Number of Parameters : {total_params}")

# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss": [], "train_dice_score": [], "val_dice_score": [], "train_jaccard_score": [], "val_jaccard_score": [], "epochs": []}
best_val_loss = float("inf")
# loop over epochs
print("[INFO] Training the network...")
start_time = time.time()
for e in tqdm(range(config.num_epochs)):
    model.train()
    
    train_loss = 0
    train_dice_scores = []
    train_jaccard_scores = []
    # loop over the training set

    for i, data in enumerate(train_loader):
        x, y = data
        # print(x)
        # print(y)
        x, y = x.to(config.device), y.to(config.device)

        optimizer.zero_grad()
        
        d0, d1, d2, d3, d4 = model(x)
        # loss1, loss2 = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, y)
        pred = d0
        # print(pred)
        # print(f"{len(x), len(pred)}")
        # print(f"{len(y)}")
        loss1 = criterion1(pred, y) 
        loss2 = criterion2(nn.Sigmoid()(pred), y)
        loss = (loss1 + loss2)/2
        loss.backward()
        optimizer.step()
        
        # Accumulate the train loss
        train_loss += loss.item() * 1.0
        
        pred = nn.Sigmoid()(pred)
        train_dice_scores.append(dice_score(y, pred).item())
        train_jaccard_scores.append(jaccard_score(y, pred).item())
        
    # Calculate train loss
    train_loss /= len(train_loader)
    train_dice_score = np.mean(train_dice_scores)
    train_jaccard_score = np.mean(train_jaccard_scores)
    
    val_loss = 0    
    val_dice_scores = [] 
    val_jaccard_scores = []
    
    model.eval()
    with torch.no_grad(): 
        for i, data in enumerate(val_loader):
            x, y = data
            x, y = x.to(config.device), y.to(config.device)
            
            
            d0, d1, d2, d3, d4 = model(x)
            pred = d0
            loss = criterion2(nn.Sigmoid()(pred), y)
            
            # Accumulate the validation loss
            val_loss += loss.item() * 1
            
            pred = nn.Sigmoid()(pred)
            
            # Accumulate the val dice scores and jaccard scores
            val_dice_scores.append(dice_score(y, pred).item())
            val_jaccard_scores.append(jaccard_score(y, pred).item())

    # Calculate validation loss
    val_loss /= len(val_loader)
    val_dice_score = np.mean(val_dice_scores)
    val_jaccard_score = np.mean(val_jaccard_scores)
    
    # update our training history
    H["train_loss"].append(train_loss)
    H["val_loss"].append(val_loss)
    H["train_dice_score"].append(train_dice_score)
    H["train_jaccard_score"].append(train_jaccard_score)
    H["val_dice_score"].append(val_dice_score)
    H["val_jaccard_score"].append(val_jaccard_score)
    H["epochs"].append(e + 1)
    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.num_epochs))
    print("Train Loss: {:.4f}, Validation Loss: {:.4f}, Train Dice Score: {:.4f}. Validation Dice Score: {:.4f}".format(
    train_loss, val_loss, train_dice_score, val_dice_score))
    
    # serialize the model to disk
    if e % 10 == 0:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        MODEL_PATH = config.architecture_name + " Epochs: {}, Date: {}.pth".format(e, current_datetime)
        torch.save(model.state_dict(), os.path.join(f"{config.output_path}/models/", MODEL_PATH))
        
    if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(f"{config.output_path}/models/", "cryosegnet_best_val_loss.pth"))

# display the total time needed to perform the training
end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    end_time - start_time))