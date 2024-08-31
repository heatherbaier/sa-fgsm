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

from adp_r18 import *

device = "cuda"

# # Load your trained model
model = torch.load("./models/central_asia_v1/kfold0/most_recent_epoch181.torch")#
# model.eval()  # Set model to evaluation mode

def resnet18(num_classes = 1000, normalize = False, use_means = False):
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, normalize = normalize, use_means = use_means)

weights = model["model_state_dict"]
geoconv = resnet18(num_classes=1, normalize = False, use_means = False)
geoconv.load_state_dict(weights)
geoconv.eval();
geoconv = geoconv.to(device);

def apply_transforms(stats):
    """Define transformations for training and validation data."""
    transforms_dict = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(stats["global"]["mean"], stats["global"]["std"])
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats["global"]["mean"], stats["global"]["std"])
        ])
    }
    return transforms_dict

#     Load transformation stats
with open("/sciclone/geograd/Heather/c1/transform_stats_sc.json", "r") as f:
    tstats = json.load(f)    
ts = apply_transforms(tstats)


def load_image(image_path):
    if ".ipynb" not in image_path:
        image = Image.fromarray(np.array(Image.open(image_path).convert("RGB"))[0:224, 0:224, :])
        image = ts["test"](image)
        return image

# Define the loss function
loss_fn = nn.L1Loss()

df = pd.read_csv("./central_asia_temp.csv")
df["iso"] = df["name"].str.split("/").str[10].str[0:2]
df.head()

samp_df = df
imnames = samp_df["name"].to_list()
labels = samp_df["label"].to_list()

coords_path = "/sciclone/geograd/heather_data/ti/data/central_africa_coords_target.json"
with open(coords_path, "r") as file:
    geo_info = json.load(file)

ys_path = "/sciclone/geograd/heather_data/ti/data/central_africa_ys_target.json"
with open(ys_path, "r") as file:
    ys = json.load(file)


def reverse_fgsm_attack(image, epsilon, data_grad):
    """
    Perform the reverse of FGSM by subtracting the sign of the gradients from the original image.

    :param image: Original input image
    :param epsilon: Perturbation magnitude
    :param data_grad: Gradient of the loss with respect to the input image
    :return: Perturbed image
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image in the opposite direction of FGSM
    perturbed_image = image - epsilon * sign_data_grad
    # Clamp the perturbed image to maintain the valid image range (e.g., [0, 1])
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image



def find_optimal_epsilon_l1(image, geo_tensor, true_label, model, loss_fn):
    """
    Find the optimal epsilon for reversing FGSM by dynamically adjusting epsilon based on model error using L1 loss.

    :param image: Input satellite image
    :param true_label: True wealth label
    :param model: Pre-trained model
    :param loss_fn: Loss function (e.g., L1Loss)
    :return: Optimal epsilon value
    """
    # Initialize variables
    epsilon = 0.01  # Starting epsilon
    best_epsilon = epsilon
    best_performance = float('inf')  # Initialize with a high value for comparison
    no_decrease_count = 0  # Track number of non-decreasing steps
    jump_count = 0  # Track the number of 0.1 jumps

    while jump_count < 3:
        # Apply reverse FGSM attack with current epsilon
        image.requires_grad = True
        output = model(image, geo_tensor)
        loss = loss_fn(output, true_label)
        model.zero_grad()
        loss.backward()
        data_grad = image.grad.data
        perturbed_image = reverse_fgsm_attack(image, epsilon, data_grad)

        # Evaluate the model on the perturbed image
        new_output = model(perturbed_image, geo_tensor)
        new_loss = loss_fn(new_output, true_label).item()

        # Check if the new loss is better (lower)
        if new_loss < best_performance:
            best_performance = new_loss
            best_epsilon = epsilon
            no_decrease_count = 0  # Reset the counter since we found a better epsilon
        else:
            no_decrease_count += 1  # Increment the counter if no improvement

        # Check if we need to jump
        if no_decrease_count >= 7:  # Increase the threshold for jumps
            epsilon += 0.1  # Jump by 0.1
            jump_count += 1
            no_decrease_count = 0  # Reset the counter for the next set of steps
        else:
            epsilon += 0.01  # Increment epsilon by 0.01

    return best_epsilon, best_performance


epsilon = 0.1 # Define a small epsilon value for the perturbation

neg, pos = 0, 0
# # Apply FGSM attack on each image
perturbed_images = []

names, best_eps = [], []

for c, (imname_full, label) in enumerate(zip(imnames, labels)):

    try:
        
        imname = imname_full[17:]
                
        image = load_image(imname)
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        
        label = torch.tensor(label).view(-1, 1).to(device)
        lon, lat = geo_info[imname_full]
        geo_tensor = torch.tensor([lon, lat], dtype=torch.float).to(device)
        
        # Enable gradient calculation
        image.requires_grad = True
        
        best_ep, best_perf = find_optimal_epsilon_l1(image, geo_tensor, label, geoconv, loss_fn)
    
        print(imname, best_ep, best_perf)
    
        names.append(imname_full)
        best_eps.append(best_ep)

    except Exception as e:
        pass
        # print(e)

    if c % 500 == 0:
        out_df = pd.DataFrame([names, best_eps]).T
        out_df.columns = "names", "best_eps"
        out_df.to_csv("/sciclone/home/hmbaier/opt_epsilon.csv", index = False)


out_df = pd.DataFrame([names, best_eps]).T
out_df.columns = "names", "best_eps"
out_df.to_csv("/sciclone/home/hmbaier/opt_epsilon_final.csv", index = False)
out_df