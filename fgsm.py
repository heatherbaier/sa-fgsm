from torchvision import models, transforms
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import random
import torch
import json
import copy
import time
import os

from fgsm_utils import *
from adp_r18 import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('region', type = str)
    parser.add_argument('model_name', type = str, help = "can be one of: SpA, DeepAll, GeoConv, FC")
    parser.add_argument('best_epoch', type = int)
    parser.add_argument('--version', required = False, default = 1, type = int)

    
    args = parser.parse_args() 

    # Load your trained model
    model = torch.load(f"./models/{args.region}_v{args.version}/kfold0/most_recent_epoch{args.best_epoch}.torch")#
    weights = model["model_state_dict"]
    
    geoconv = resnet18(num_classes=1, normalize = False, use_means = False)
    geoconv.load_state_dict(weights)
    geoconv.eval();
    geoconv = geoconv.to(device);

    # Define the loss function
    loss_fn = nn.L1Loss()
    
    with open(f"/sciclone/geograd/heather_data/ti/data/{args.region}_ys_target.json", "r") as f:
        df = json.load(f)   

    imnames = list(df.keys())
    labels = list(df.values())
    
    coords_path = f"/sciclone/geograd/heather_data/ti/data/{args.region}_coords_target.json"
    with open(coords_path, "r") as file:
        # with open(f"{prefix}/sciclone/geograd/Heather/c1/data/clean/coords.json", "r") as file:
        geo_info = json.load(file)
    
    epsilon = 0.01  # Define a small epsilon value for the perturbation
    
    neg, pos = 0, 0
    # # Apply FGSM attack on each image
    perturbed_images = []
    
    names, og_losses, pert_losses, og_pred, pert_pred = [], [], [], [], []
    
    for c, (imname_full, label) in enumerate(zip(imnames, labels)):
    
        try:
    
            # print(imname_full)

            imname = imname_full[17:]

            label = torch.tensor(label).view(-1, 1).to(device)
            lon, lat = geo_info[imname_full]
            geo_tensor = torch.tensor([lon, lat], dtype=torch.float).to(device)
            
                    
            image = load_image(imname)
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            
            # Enable gradient calculation
            image.requires_grad = True
            
            # Forward pass
            output = geoconv(image, geo_tensor)
            
            # Calculate the loss
            loss = loss_fn(output, label)
            
            # Zero all existing gradients
            geoconv.zero_grad()
            
            # Backward pass to calculate gradients
            loss.backward()
            
            # Collect the gradients of the image
            data_grad = image.grad.data
            
            # Call FGSM to create the perturbed image
            perturbed_image = reverse_fgsm_attack(image, epsilon, data_grad)
            
            # Store the perturbed image
            # perturbed_images.append(perturbed_image)
        
            # Convert list to tensor
            # perturbed_images = torch.cat(perturbed_images)
    
            perturbed_outputs = geoconv(perturbed_image, geo_tensor)
            perturbed_loss = loss_fn(perturbed_outputs, label)
    
            names.append(imname_full)
            og_losses.append(loss.item())
            pert_losses.append(perturbed_loss.item())
            og_pred.append(output.item())
            pert_pred.append(perturbed_outputs.item())        
    
            # print("OG Loss: ", loss.item(), "OG Prediction: ", output.item(), "Pert. Pred: ", perturbed_outputs.item(), "Pert. Loss: ", perturbed_loss.item())
            # print("Change in loss: ", loss.item() - perturbed_loss.item(), "\n")
    
            if (loss.item() - perturbed_loss.item()) < 0:
                neg += 1
            else:
                pos += 1
    
            if c % 500 == 0:
                out_df = pd.DataFrame([names, og_losses, pert_losses, og_pred, pert_pred]).T
                out_df.columns = "names", "og_losses", "pert_losses", "og_pred", "pert_pred"
                out_df.to_csv(f"/sciclone/geograd/heather_data/ti/fgsm_results/{args.region}_pert_losses.csv", index = False)

            print(c, len(imnames), end = "\r")
        
        except Exception as e:
            # print(e)
            pass
    
        print("Decreased Acc: ", neg, "Increased Acc: ", pos, end = "\r")
    
    out_df = pd.DataFrame([names, og_losses, pert_losses, og_pred, pert_pred]).T
    out_df.columns = "names", "og_losses", "pert_losses", "og_pred", "pert_pred"
    out_df.to_csv(f"/sciclone/geograd/heather_data/ti/fgsm_results/{args.region}_pert_losses_final.csv", index = False)
    
