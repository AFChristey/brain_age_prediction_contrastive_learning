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


# Define the MMD function
def mmd_rbf(X, Y, gamma=1.0):
    XX = torch.exp(-gamma * torch.cdist(X, X, p=2).pow(2))
    YY = torch.exp(-gamma * torch.cdist(Y, Y, p=2).pow(2))
    XY = torch.exp(-gamma * torch.cdist(X, Y, p=2).pow(2))
    return XX.mean() + YY.mean() - 2 * XY.mean()

# # List of datasets
# datasets = {
#     "Dynamic_DR": ("/rds/user/afc53/hpc-work/saved_features/Dynamic_DR/features_before_reduction_epoch_50.npy",
#                    "/rds/user/afc53/hpc-work/saved_features/Dynamic_DR/metadata_epoch_50.npy"),
#     "Dynamic_Stiffness": ("/rds/user/afc53/hpc-work/saved_features/Dynamic_Stiffness/features_before_reduction_epoch_50.npy",
#                           "/rds/user/afc53/hpc-work/saved_features/Dynamic_Stiffness/metadata_epoch_50.npy"),
#     "Dynamic_T1": ("/rds/user/afc53/hpc-work/saved_features/Dynamic_T1/features_before_reduction_epoch_50.npy",
#                    "/rds/user/afc53/hpc-work/saved_features/Dynamic_T1/metadata_epoch_50.npy"),
#     "Exponential_DR": ("/rds/user/afc53/hpc-work/saved_features/Exponential_DR/features_before_reduction_epoch_300.npy",
#                         "/rds/user/afc53/hpc-work/saved_features/Exponential_DR/metadata_epoch_300.npy"),
#     "Exponential_Stiffness": ("/rds/user/afc53/hpc-work/saved_features/Exponential_Stiffness/features_before_reduction_epoch_300.npy",
#                               "/rds/user/afc53/hpc-work/saved_features/Exponential_Stiffness/metadata_epoch_300.npy"),
#     "Exponential_T1": ("/rds/user/afc53/hpc-work/saved_features/Exponential_T1/features_before_reduction_epoch_300.npy",
#                         "/rds/user/afc53/hpc-work/saved_features/Exponential_T1/metadata_epoch_300.npy"),
#     "RankN_DR": ("/rds/user/afc53/hpc-work/saved_features/RankN_DR/features_before_reduction_epoch_100.npy",
#                  "/rds/user/afc53/hpc-work/saved_features/RankN_DR/metadata_epoch_100.npy"),
#     "RankN_Stiffness": ("/rds/user/afc53/hpc-work/saved_features/RankN_Stiffness/features_before_reduction_epoch_100.npy",
#                         "/rds/user/afc53/hpc-work/saved_features/RankN_Stiffness/metadata_epoch_100.npy"),
#     "RankN_T1": ("/rds/user/afc53/hpc-work/saved_features/RankN_T1/features_before_reduction_epoch_100.npy",
#                  "/rds/user/afc53/hpc-work/saved_features/RankN_T1/metadata_epoch_100.npy"),
# }

# List of datasets
datasets = {
    "Dynamic_BHB": ("/rds/user/afc53/hpc-work/saved_features/Dynamic_BHB/features_before_reduction_epoch_0_dynamic.npy",
                   "/rds/user/afc53/hpc-work/saved_features/Dynamic_BHB/metadata_epoch_0_dynamic.npy"),
    "Exponential_BHB": ("/rds/user/afc53/hpc-work/saved_features/Exponential_BHB/features_before_reduction_epoch_0.npy",
                   "/rds/user/afc53/hpc-work/saved_features/Exponential_BHB/metadata_epoch_0.npy"),
    "RankN_BHB": ("/rds/user/afc53/hpc-work/saved_features/RankN_BHB/features_before_reduction_epoch_0_supcon.npy",
                   "/rds/user/afc53/hpc-work/saved_features/RankN_BHB/metadata_epoch_0_supcon.npy")
}



results = {}

# Loop through datasets
for name, (features_path, metadata_path) in datasets.items():
    features = np.load(features_path)
    metadata = np.load(metadata_path)
    
    # # Encode site labels
    # label_encoder = LabelEncoder()
    # site_labels = label_encoder.fit_transform(metadata)
    
    # Silhouette Score
    sil_score = silhouette_score(features, site_labels)
    
    # Mutual Information
    mi_score = mutual_info_classif(features, site_labels).mean()
    
    # Convert to tensors
    site_labels = torch.tensor(site_labels, dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float32)
    
    # Unique site labels
    unique_sites = torch.unique(site_labels)
    
    # MMD Calculation
    mmd_scores = {}
    for site_A, site_B in itertools.combinations(unique_sites, 2):
        features_A = features[site_labels == site_A]
        features_B = features[site_labels == site_B]

        if features_A.shape[0] > 0 and features_B.shape[0] > 0:
            mmd_score = mmd_rbf(features_A, features_B)
            mmd_scores[(site_A.item(), site_B.item())] = mmd_score.item()

    # Mean MMD
    mean_mmd_score = np.mean(list(mmd_scores.values())) if mmd_scores else 0

    results[name] = {
        "Silhouette Score": sil_score,
        "Mutual Information Score": mi_score,
        "MMD Scores": mmd_scores,
        "Mean MMD Score": mean_mmd_score
    }

import pandas as pd

# Convert results to DataFrame for better visualization
summary_data = []
for dataset, metrics in results.items():
    for (site_A, site_B), mmd_score in metrics["MMD Scores"].items():
        summary_data.append({
            "Dataset": dataset,
            "Silhouette Score": metrics["Silhouette Score"],
            "Mutual Information Score": metrics["Mutual Information Score"],
            "Site A": site_A,
            "Site B": site_B,
            "MMD Score": mmd_score,
            "Mean MMD Score": metrics["Mean MMD Score"]
        })

summary_df = pd.DataFrame(summary_data)



# Save the DataFrame to a CSV file in the current directory
csv_file_path = '/home/afc53/images/confound_metrics_results.csv'
summary_df.to_csv(csv_file_path, index=False)












# features_dynamic_dr = np.load("/rds/user/afc53/hpc-work/saved_features/Dynamic_DR/features_before_reduction_epoch_50.npy")
# metadata_dynamic_dr = np.load("/rds/user/afc53/hpc-work/saved_features/Dynamic_DR/metadata_epoch_50.npy")
# features_dynamic_stiff = np.load("/rds/user/afc53/hpc-work/saved_features/Dynamic_Stiffness/features_before_reduction_epoch_50.npy")
# metadata_dynamic_stiff = np.load("/rds/user/afc53/hpc-work/saved_features/Dynamic_Stiffness/metadata_epoch_50.npy")
# features_dynamic_t1 = np.load("/rds/user/afc53/hpc-work/saved_features/Dynamic_T1/features_before_reduction_epoch_50.npy")
# metadata_dynamic_t1 = np.load("/rds/user/afc53/hpc-work/saved_features/Dynamic_T1/metadata_epoch_50.npy")


# features_exponential_dr = np.load("/rds/user/afc53/hpc-work/saved_features/Exponential_DR/features_before_reduction_epoch_50.npy")
# metadata_exponential_dr = np.load("/rds/user/afc53/hpc-work/saved_features/Exponential_DR/metadata_epoch_50.npy")
# features_exponential_stiff = np.load("/rds/user/afc53/hpc-work/saved_features/Exponential_Stiffness/features_before_reduction_epoch_50.npy")
# metadata_exponential_stiff = np.load("/rds/user/afc53/hpc-work/saved_features/Exponential_Stiffness/metadata_epoch_50.npy")
# features_exponential_t1 = np.load("/rds/user/afc53/hpc-work/saved_features/Exponential_T1/features_before_reduction_epoch_50.npy")
# metadata_exponential_t1 = np.load("/rds/user/afc53/hpc-work/saved_features/Exponential_T1/metadata_epoch_50.npy")


# features_rankN_dr = np.load("/rds/user/afc53/hpc-work/saved_features/RankN_DR/features_before_reduction_epoch_50.npy")
# metadata_rankN_dr = np.load("/rds/user/afc53/hpc-work/saved_features/RankN_DR/metadata_epoch_50.npy")
# features_rankN_stiff = np.load("/rds/user/afc53/hpc-work/saved_features/RankN_Stiffness/features_before_reduction_epoch_50.npy")
# metadata_rankN_stiff = np.load("/rds/user/afc53/hpc-work/saved_features/RankN_Stiffness/metadata_epoch_50.npy")
# features_rankN_t1 = np.load("/rds/user/afc53/hpc-work/saved_features/RankN_T1/features_before_reduction_epoch_50.npy")
# metadata_rankN_t1 = np.load("/rds/user/afc53/hpc-work/saved_features/RankN_T1/metadata_epoch_50.npy")


# def mmd_rbf(X, Y, gamma=1.0):
#     """Compute Maximum Mean Discrepancy (MMD) using an RBF kernel (Gaussian kernel)."""
#     XX = torch.exp(-gamma * torch.cdist(X, X, p=2).pow(2))  # Kernel K(X, X)
#     YY = torch.exp(-gamma * torch.cdist(Y, Y, p=2).pow(2))  # Kernel K(Y, Y)
#     XY = torch.exp(-gamma * torch.cdist(X, Y, p=2).pow(2))  # Kernel K(X, Y)
    
#     return XX.mean() + YY.mean() - 2 * XY.mean()


# # features = [[0.03586222, -0.00067997, -0.01195611, -0.02239724, 0.1086802, -0.00415948], 
# #             [-0.02256612, 0.03379672, -0.05384277, 0.04146272, 0.05478617, 0.0526447], 
# #             [ 0.01575743, 0.01593883, -0.02351342, -0.00440824, 0.099177, 0.01441356],
# #             [-0.0191126, 0.02508227, -0.05166087, 0.03427609, 0.05926874, 0.04427647]]
# # metadata = ['ATLAS', 'MIMS', 'NOVA', 'ATLAS']

# site_labels = metadata

# label_encoder = LabelEncoder()
# site_labels = label_encoder.fit_transform(site_labels)

# # Compute silhouette score
# sil_score = silhouette_score(features, site_labels)
# print(f"Silhouette Score: {sil_score}")

# # Compute Mutual Information
# mi_score = mutual_info_classif(features, site_labels)
# # Average MI over all feature dimensions
# mi_score_mean = mi_score.mean()
# print(f"Mutual Information Score: {mi_score_mean}")

# site_labels = torch.tensor(site_labels, dtype=torch.long)

# # Unique site labels (0 to 5)
# unique_sites = torch.arange(6)  # Since sites are encoded as 0,1,2,3,4,5
# # print(unique_sites)
# # Dictionary to store MMD results
# mmd_scores = {}

# for site_A, site_B in itertools.combinations(unique_sites, 2):
#     features = torch.tensor(features, dtype=torch.float32)
#     features_A = features[site_labels == site_A]
#     features_B = features[site_labels == site_B]

#     features_A = torch.tensor(features_A, dtype=torch.float32)
#     features_B = torch.tensor(features_B, dtype=torch.float32)

#     # Only compute MMD if both sites have samples
#     if features_A.shape[0] > 0 and features_B.shape[0] > 0:
#         mmd_score = mmd_rbf(features_A, features_B)
#         mmd_scores[(site_A.item(), site_B.item())] = mmd_score.item()

# # Print all MMD results
# for (site_A, site_B), score in mmd_scores.items():
#     print(f"MMD Score between Site {site_A} and Site {site_B}: {score}")

# # Calculate and print the mean MMD score
# mean_mmd_score = np.mean(list(mmd_scores.values()))
# print(f"Mean MMD Score across all site pairs: {mean_mmd_score}")


