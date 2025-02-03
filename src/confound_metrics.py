# import datetime
# import math
# import os
# from random import gauss
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.utils.data
# import torchvision
# import argparse
# import models
# import losses
# import time
# # import wandb
# import torch.utils.tensorboard

# from torch import nn
# from torchvision import transforms
# from torchvision import datasets
# from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
# from util import warmup_learning_rate, adjust_learning_rate
# # from data import FeatureExtractor, OpenBHB, bin_age
# from data import OpenBHB, bin_age
# from data.transforms import Crop, Pad, Cutout
# #from main_mse import get_transforms
# from util import get_transforms

# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import matplotlib.pyplot as plt
# import umap
# import seaborn as sns
# import pandas as pd

# from sklearn.manifold import TSNE





# def compute_site_ba(features, labels):
#     site_estimator = models.SiteEstimator()

#     print("Training site estimator")
#     train_X, train_y = gather_site_feats(model, train_loader, opts)
#     # print(train_X.shape)
#     # print(train_y)
#     ba_train = site_estimator.fit(train_X, train_y)

#     print("Computing BA")
#     test_X, test_y = gather_site_feats(model, test_loader, opts)
#     # ext_X, ext_y = gather_site_feats(model, test_ext, opts)
#     ba_test = site_estimator.score(test_X, test_y)
#     # ba_ext = site_estimator.score(ext_X, ext_y)



# mae_train, mae_test = compute_age_mae(features, labels)
# print("Age MAE:", mae_train, mae_test)



# ba_train, ba_test = compute_site_ba(features, labels)
# print("Site BA:", ba_train, ba_test)