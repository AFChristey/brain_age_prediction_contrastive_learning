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

site_labels = metadata[1]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform labels to numbers
site_labels = label_encoder.fit_transform(site_labels)


# Compute silhouette score
sil_score = silhouette_score(features, site_labels)
print(f"Silhouette Score: {sil_score}")

# Compute Mutual Information
mi_score = mutual_info_classif(features, site_labels)
# Average MI over all feature dimensions
mi_score_mean = mi_score.mean()
print(f"Mutual Information Score: {mi_score_mean}")




