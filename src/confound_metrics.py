import datetime
import math
import os
from random import gauss
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import argparse
import models
import losses
import time
# import wandb
import torch.utils.tensorboard
import itertools

from torch import nn
from torchvision import transforms
from torchvision import datasets
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
# from data import FeatureExtractor, OpenBHB, bin_age
from data import OpenBHB, bin_age
from data.transforms import Crop, Pad, Cutout
#from main_mse import get_transforms
from util import get_transforms

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib.pyplot as plt
import umap
import seaborn as sns
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder



features = np.load("/rds/user/afc53/hpc-work/saved_features/Dynamic_DR/features_before_reduction_epoch_50.npy")
metadata = np.load("/rds/user/afc53/hpc-work/saved_features/Dynamic_DR/metadata_epoch_50.npy")

print(metadata)

site_labels = metadata

print(site_labels.shape)
print(features.shape)
print(site_labels)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform labels to numbers
site_labels = label_encoder.fit_transform(site_labels)

print(site_labels)

# Compute silhouette score
sil_score = silhouette_score(features, site_labels)
print(f"Silhouette Score: {sil_score}")

# Compute Mutual Information
mi_score = mutual_info_classif(features, site_labels)
# Average MI over all feature dimensions
mi_score_mean = mi_score.mean()
print(f"Mutual Information Score: {mi_score_mean}")


# Compute MMD:


# ATLAS
# BMI
# NOVA
# NITRC_1
# NITRC_2
# MIMS



def mmd_rbf(X, Y, gamma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) using an RBF kernel (Gaussian kernel)."""
    XX = torch.exp(-gamma * torch.cdist(X, X, p=2).pow(2))  # Kernel K(X, X)
    YY = torch.exp(-gamma * torch.cdist(Y, Y, p=2).pow(2))  # Kernel K(Y, Y)
    XY = torch.exp(-gamma * torch.cdist(X, Y, p=2).pow(2))  # Kernel K(X, Y)
    
    return XX.mean() + YY.mean() - 2 * XY.mean()


# Unique site labels (0 to 5)
unique_sites = torch.arange(6)  # Since sites are encoded as 0,1,2,3,4,5

# Dictionary to store MMD results
mmd_scores = {}

# Iterate over all unique pairs of sites (0-5)
for site_A, site_B in itertools.combinations(unique_sites, 2):
    features_A = features[site_labels == site_A]
    features_B = features[site_labels == site_B]
    
    # Only compute MMD if both sites have samples
    if features_A.shape[0] > 0 and features_B.shape[0] > 0:
        mmd_score = mmd_rbf(features_A, features_B)
        mmd_scores[(site_A.item(), site_B.item())] = mmd_score.item()

# Print all MMD results
for (site_A, site_B), score in mmd_scores.items():
    print(f"MMD Score between Site {site_A} and Site {site_B}: {score}")



