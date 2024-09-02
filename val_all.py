# Assuming the ResNet and BasicBlock classes are defined as provided in the previous messages
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import time
from sklearn.metrics import r2_score


from utils import *
from dataloader import PlanetData


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_name', type = str)
    parser.add_argument('model_name', type = str, help = "can be one of: SpA, DeepAll, GeoConv, FC")
    parser.add_argument('--eval', action='store_true', help='')
    parser.add_argument('--start_epoch', required = False, default = 0, type = int)
    parser.add_argument('--kfold', required = False, default = 0)
    parser.add_argument('--num_epochs', required = False, default = 200, type = int)
    parser.add_argument('--labels_path', required = False, default = "/sciclone/geograd/heather_data/mex_disagg/mex_subset.json")
    parser.add_argument('--transforms_path', required = False, default = "/sciclone/geograd/Heather/c1/transform_stats_sc.json")
    parser.add_argument('--coords_path', required = False, default = "/sciclone/geograd/heather_data/mex_disagg/slv_grid_coords.json")
    parser.add_argument('--device', required = False, default = "cuda")
    parser.add_argument('--postval', action='store_true', help='')
    parser.add_argument('--ts4', action='store_true', help='')
    parser.add_argument('--use_means', action='store_true', help='')
    parser.add_argument('--strict', action='store_false', help='')
    parser.add_argument('--norm_coords', action='store_true', help='')
    args = parser.parse_args()
    
    print(args)
    
    # Add prefix to file paths if running on ts4
    if args.ts4:
        transforms_path = "/rapids/notebooks" + args.transforms_path
        folder_name = "/rapids/notebooks" + args.folder_name
        labels_path = "/rapids/notebooks" + args.labels_path
    else:
        transforms_path = args.transforms_path
        folder_name = args.folder_name
        labels_path = args.labels_path     
        
    # Load transformation stats
    with open(transforms_path, "r") as f:
        tstats = json.load(f)    
    ts = apply_transforms(tstats)
        
    num_epochs = args.num_epochs
    device = args.device
    
#     print(args.start_epoch)
    
#     if args.start_epoch == 0:
    if not os.path.exists(f"{folder_name}/results/"):
        os.mkdir(f"{folder_name}/results/")
        
    # epoch = args.start_epoch

    for epoch in reversed(range(args.start_epoch, 118)):

        # epoch = 88

        print("Epoch: ", epoch)
    
        # Instantiate model
        model = construct_model(args.model_name, args.norm_coords, args.use_means)
        model.to(args.device)  
        #         model.eval();
    
        #         print(model)
    
        weights = torch.load(f"{folder_name}/most_recent_epoch{epoch}.torch")["model_state_dict"]
        model.load_state_dict(weights, strict = args.strict)
    
        if args.eval:
            model.eval();
    
        with open(labels_path, "r") as file:
            labels = json.load(file)
        labels = list(labels.keys())
        labels[0:5]
    
        with open(f"{folder_name}/val_indices_fold{args.kfold}.txt", "r") as f:
            names = f.read().splitlines()
        valnames = [labels[int(i)] for i in names]
        valnames[0:5]
    
        target_dataset = PlanetData(labels_path=labels_path, 
                                    coords_path = args.coords_path,
                                    transform=ts, 
                                    sample=0, 
                                    postval=True, 
                                    ts4 = args.ts4, 
                                    valnames = valnames)
    
        preds, labs, ns = [], [], []
        for count, (inputs, targets, coords, imname) in enumerate(target_dataset):
    
            if (args.model_name in ["SpA", "FC"]) or (args.norm_coords):
                coords = coords.unsqueeze(0)
    
        #             if args.norm_coords in ["SpA", "FC"]:
        #                 coords = coords.unsqueeze(0)                
        #                 
            if args.model_name in ["SpA", "GeoConv", "FC"]:
                output = model(inputs.unsqueeze(0).to(device), coords.to(device))
            else:
                output = model(inputs.unsqueeze(0).to(device))
    
            preds.append(output.item())
            labs.append(targets)
            ns.append(imname)
            print(count, len(target_dataset), end = "\r")
    
            if count % 5000 == 0:
                df = pd.DataFrame([preds, labs, ns]).T
                df.columns = ["pred", "label", "name"]
                df.to_csv(f"{folder_name}/results/epoch{epoch}_preds.csv", index = False)                
    
        df = pd.DataFrame([preds, labs, ns]).T
        df.columns = ["pred", "label", "name"]
        df.to_csv(f"{folder_name}/results/epoch{epoch}_preds.csv", index = False)
    
        print(r2_score(df["label"], df["pred"]))