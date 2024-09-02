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

# Load your trained model
def resnet18(num_classes = 1000, normalize = False, use_means = False):
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, normalize = normalize, use_means = use_means)


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