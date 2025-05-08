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
import wandb
import torch.utils.tensorboard

from torch import nn
from torchvision import transforms
from torchvision import datasets
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae, compute_site_ba
# from data import FeatureExtractor, OpenBHB, bin_age
from data import MREData, OpenBHB, bin_age
from data.transforms import Crop, Pad, Cutout
#from main_mse import get_transforms
# from util import get_transforms
from util import get_transforms, get_transforms_OpenBHB

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import matplotlib.pyplot as plt
import umap
import seaborn as sns
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import torch.optim.lr_scheduler as lr_scheduler

import itertools

from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif





# which_data_type = 'MRE' 
# which_data_type = 'MRE' 
is_sweeping = False

# import os
# os.environ["WANDB_MODE"] = "disabled"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--save_freq', type=int, help='save frequency', default=10)
    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/data')
    parser.add_argument('--amp', type=arg2bool, help='use amp', default=False)
    parser.add_argument('--clip_grad', type=arg2bool, help='clip gradient to prevent nan', default=False)
    parser.add_argument('--wandb_name', type=str, help='name of the wandb project', default='contrastive-brain-age-prediction')
    parser.add_argument('--error', type=str, help='_______', default='trial')

    # Model
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')

    # Optimizer
    parser.add_argument('--epochs', type=int, help='number of epochs', default=300)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs', default=10)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--optimizer', type=str, help="optimizer (adam or sgd)", choices=["adam", "sgd"], default="adam")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=5e-5)

    # Data
    parser.add_argument('--train_all', type=arg2bool, help='train on all dataset including validation (int+ext)', default=True)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'noise', 'all'], default='none')
    parser.add_argument('--noise_std', type=float, help='std for noise augmentation', default=0.05)
    parser.add_argument('--path', type=str, help='model ran on cluster or locally', choices=['local', 'cluster'], default='local')
    parser.add_argument('--loss_choice', type=str, help='which loss function is being tested', choices=['supcon', 'dynamic', 'RnC'], default='supcon')
    parser.add_argument('--modality', type=str, help='which type of data to use', choices=['OpenBHB', 'MRE'], default='OpenBHB')
    
    # Loss 
    parser.add_argument('--method', type=str, help='loss function', choices=['supcon', 'yaware', 'threshold', 'expw'], default='supcon')
    parser.add_argument('--kernel', type=str, help='Kernel function (not for supcon)', choices=['cauchy', 'gaussian', 'rbf'], default=None)
    parser.add_argument('--delta_reduction', type=str, help='use mean or sum to reduce 3d delta mask (only for method=threshold)', default='sum')
    parser.add_argument('--temp', type=float, help='loss temperature', default=0.1)
    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--sigma', type=float, help='gaussian-rbf kernel sigma / cauchy gamma', default=1)
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--n_views', type=int, help='num. of multiviews', default=2)


    parser.add_argument('--lambda_adv', type=float, help='Weight for adversarial loss', default=0.1)
    parser.add_argument('--lambda_mmd', type=float, help='Weight for MMD loss', default=0)
    parser.add_argument('--grl_layer', type=float, help='turn on or off grl layer', default=0)
    parser.add_argument('--lambda_val', type=float, help='strength of grl layer', default=0)
    parser.add_argument('--confound_loss', type=str, help='loss chosen for removing confound effect', choices=['basic', 'classification', 'mmd', 'class+mmd', 'classificationGRL', 'coral', 'hsic', 'dsn'], default='basic')
    parser.add_argument('--lambda_coral', type=float, help='Weight for Coral loss', default=0)
    parser.add_argument('--lambda_hsic', type=float, help='Weight for HSIC loss', default=0)
    parser.add_argument('--lambda_recon', type=float, help='Weight for reconstruction loss', default=0)
    parser.add_argument('--lambda_diff', type=float, help='Weight for difference loss', default=0)





    # RnCLoss Parameters
    parser.add_argument('--temp_RNC', type=float, default=2, help='temperature for RnC loss')
    parser.add_argument('--label_diff', type=str, default='l1', choices=['l1'], help='label distance function')
    parser.add_argument('--feature_sim', type=str, default='l2', choices=['l2'], help='feature similarity function')


    # dynamic loss hyperparameter for new modifications
    parser.add_argument('--NN_nb_step_size', type=int, help='step size for NN_nb', default=0)
    parser.add_argument('--end_NN_nb', type=int, help='label type', default=4)
    parser.add_argument('--NN_nb_selection', type=str, help='selection method for NN_nb',
                        choices=['euclidean', 'similarity', 'manhattan', 'chebyshev', 'no'], default='similarity')


    opts = parser.parse_args()

    if opts.batch_size > 256:
        print("Forcing warm")
        opts.warm = True

    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))

    if opts.warm:
        opts.warmup_from = 0.01
        opts.warm_epochs = 10
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (opts.lr_decay_rate ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (
                    1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
            opts.warmup_to = opts.lr

    if opts.method == 'supcon':
        print('method == supcon, binning age')
        opts.label = 'bin'
    else:
        print('method != supcon, using real age value')
        opts.label = 'cont'

    if opts.method == 'supcon' and opts.kernel is not None:
        print('Invalid kernel for supcon')
        exit(0)
    
    if opts.method != 'supcon' and opts.kernel is None:
        print('Kernel cannot be None for method != supcon')
        exit(1)
    
    if opts.model == 'densenet121':
        opts.n_views = 1
    
    print(opts.path)     

    return opts

def load_data(opts):

    if opts.modality == 'OpenBHB':
        print('getting transforms')
        # T_train, T_test = get_transforms_OpenBHB(opts)
        T_train, T_test = get_transforms(opts)
        T_train = NViewTransform(T_train, opts.n_views)

        print('transformed data')

        start_time = time.time()


        train_dataset = OpenBHB(train=True, transform=T_train, path=opts.path)
        train_dataset.norm()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
        train_time = time.time() - start_time
        print(f"Time to load training dataset: {train_time:.2f} seconds")

        start_time = time.time()

        train_dataset_score = OpenBHB(train=True, transform=T_train, path=opts.path)
        train_dataset_score.norm()
        train_loader_score = torch.utils.data.DataLoader(train_dataset_score, batch_size=opts.batch_size, shuffle=False)

        train_score_time = time.time() - start_time
        print(f"Time to load training dataset (score): {train_score_time:.2f} seconds")

        start_time = time.time()

        test_dataset = OpenBHB(train=False, transform=T_test, path=opts.path)
        test_dataset.norm()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)

        test_time = time.time() - start_time
        print(f"Time to load test dataset: {test_time:.2f} seconds")



    else:
        T_train, T_test = get_transforms(opts)
        T_train = NViewTransform(T_train, opts.n_views)

        print("reading data")
        train_dataset = MREData(modality='stiffness', train=True, transform=T_train, label=opts.label, path=opts.path, fold=0)

        # print('HELLO')
        train_dataset.norm()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)

        train_dataset_score = MREData(modality='stiffness', train=True, transform=T_train, label=opts.label, path=opts.path, fold=0)

        train_dataset_score.norm()
        # print('HELLO')

        print('still reading')


        train_loader_score = torch.utils.data.DataLoader(train_dataset_score, batch_size=opts.batch_size, shuffle=False)

        test_dataset = MREData(modality='stiffness', train=False, transform=T_test, path=opts.path, fold=0)

        test_dataset.norm()

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)

        print("Done reading")
        # print(train_dataset.shape)
    
    # print('TEST LOADER SIZE')
    # print(test_internal)
    print(len(test_loader.dataset))
    return train_loader, train_loader_score, test_loader

def load_model(opts):
    if 'resnet' in opts.model:
        if opts.modality == "OpenBHB":
            if opts.confound_loss == "dsn":
                # model = models.SupConResNet(opts.model, feat_dim=128, num_sites=70, grl_layer=opts.grl_layer, lambda_val=opts.lambda_val, modality=opts.modality)
                model = models.DSN(modality=opts.modality, grl_layer=opts.grl_layer)
            else:
                model = models.SupConResNet(opts.model, feat_dim=128, num_sites=70, grl_layer=opts.grl_layer, lambda_val=opts.lambda_val, modality=opts.modality)
            # model = models.SupConResNet(opts.model, feat_dim=128, num_sites=70)
        else:
            if opts.confound_loss == "dsn":
                model = models.DSN(modality=opts.modality)           
            else:
                model = models.SupConResNet(opts.model, feat_dim=128, grl_layer=opts.grl_layer, lambda_val=opts.lambda_val, modality=opts.modality)
            # model = models.SupConResNet(opts.model, feat_dim=128)
    elif 'alexnet' in opts.model:
        model = models.SupConAlexNet(feat_dim=128)
    elif 'densenet121' in opts.model:
        model = models.SupConDenseNet(feat_dim=128)
    
    else:
        raise ValueError("Unknown model", opts.model)

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    # print(opts.device)

    if opts.path == "local":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = model.to(device)
    # CHANGED Al   
    else: 
        model = model.to(opts.device)



    def gaussian_kernel(x):
        x = x - x.T
        return torch.exp(-(x**2) / (2*(opts.sigma**2))) / (math.sqrt(2*torch.pi)*opts.sigma)
    
    def rbf(x):
        x = x - x.T
        return torch.exp(-(x**2)/(2*(opts.sigma**2)))
    
    def cauchy(x):
        x = x - x.T
        return  1. / (opts.sigma*(x**2) + 1)

    kernels = {
        'none': None,
        'cauchy': cauchy,
        'gaussian': gaussian_kernel,
        'rbf': rbf
    }

    if opts.loss_choice == "supcon":
        infonce = losses.KernelizedSupCon(method=opts.method, temperature=opts.temp, 
                                          kernel=kernels[opts.kernel], delta_reduction=opts.delta_reduction)
        
    elif opts.loss_choice == "dynamic":
        infonce = losses.DynLocRep_loss(method=opts.method, temperature=opts.temp, kernel=kernels[opts.kernel],
                                            delta_reduction=opts.delta_reduction, epochs=opts.epochs,
                                            NN_nb_step_size=opts.NN_nb_step_size, end_NN_nb=opts.end_NN_nb,
                                            NN_nb_selection=opts.NN_nb_selection)
        
    elif opts.loss_choice == "RnC":
        infonce = losses.RnCLoss(temperature=opts.temp, label_diff=opts.label_diff, feature_sim=opts.feature_sim)

    # print("FEFEGEE  EFE FE  GNRGIRNGRIGNRIGRGNRGINGRIGNRGNRIGRIGNRGIRGRGNRGRIN")            

    infonce = infonce.to(opts.device)
    
    return model, infonce

def load_optimizer(model, opts):
    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, 
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    else:
        # optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))

    return optimizer


# def mmd(X, Y, gamma=1.0):
#     # def mmd_rbf(X, Y, gamma=1.0):
#     XX = torch.exp(-gamma * torch.cdist(X, X, p=2).pow(2))
#     YY = torch.exp(-gamma * torch.cdist(Y, Y, p=2).pow(2))
#     XY = torch.exp(-gamma * torch.cdist(X, Y, p=2).pow(2))
#     return XX.mean() + YY.mean() - 2 * XY.mean()

def mmd(X, Y):
    bandwidth_range=[0.2, 0.5, 0.9, 1.3]

    XX, YY, XY = 0, 0, 0 

    for gamma in bandwidth_range:
        XX += torch.exp(-gamma * torch.cdist(X, X, p=2).pow(2))
        YY += torch.exp(-gamma * torch.cdist(Y, Y, p=2).pow(2))
        XY += torch.exp(-gamma * torch.cdist(X, Y, p=2).pow(2))

    XX /= len(bandwidth_range)
    YY /= len(bandwidth_range)
    XY /= len(bandwidth_range)

    return XX.mean() + YY.mean() - 2 * XY.mean()

def mmd_calculator(opts, projected, site_labels):

    unique_sites = torch.unique(site_labels)

    mmd_values = {}
    for site_A, site_B in itertools.combinations(unique_sites, 2):
        features_A = projected[site_labels == site_A]
        features_B = projected[site_labels == site_B]

        if features_A.shape[0] > 0 and features_B.shape[0] > 0:
            mmd_score = mmd(features_A, features_B)
            mmd_values[(site_A.item(), site_B.item())] = mmd_score.item()

    # Mean MMD
    mmd_value = np.mean(list(mmd_values.values())) if mmd_values else 0



    return mmd_value


def coral(X, Y):
    """
    CORAL loss between features X and Y from two domains (e.g., sites).
    """
    d = X.size(1)

    # Center the features
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)

    # Compute covariance matrices
    cov_X = X_c.T @ X_c / (X.size(0) - 1)
    cov_Y = Y_c.T @ Y_c / (Y.size(0) - 1)

    # Frobenius norm between covariance matrices
    loss = torch.mean((cov_X - cov_Y) ** 2)

    return loss / (4 * d * d)

def coral_calculator(opts, projected, site_labels):
    """
    Computes average CORAL loss across all unique site/domain pairs.
    """
    unique_sites = torch.unique(site_labels)
    coral_values = {}

    for site_A, site_B in itertools.combinations(unique_sites, 2):
        features_A = projected[site_labels == site_A]
        features_B = projected[site_labels == site_B]

        if features_A.shape[0] > 1 and features_B.shape[0] > 1:
            coral_score = coral(features_A, features_B)
            coral_values[(site_A.item(), site_B.item())] = coral_score.item()

    coral_value = np.mean(list(coral_values.values())) if coral_values else 0
    return coral_value



def hsic_calculator(X, Y, sigma=1.0):
    """Compute empirical HSIC with RBF kernels."""
    def rbf_kernel(x, sigma):
        dist = torch.cdist(x, x).pow(2)
        return torch.exp(-dist / (2 * sigma**2))

    n = X.size(0)
    H = torch.eye(n, device=X.device) - (1. / n) * torch.ones(n, n, device=X.device)

    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)

    HKH = H @ K @ H
    HLH = H @ L @ H

    hsic = torch.trace(HKH @ HLH) / ((n - 1) ** 2)
    return hsic

def train(train_loader, model, infonce, optimizer, opts, epoch):
    # lambda_adv = 0.35  # Weight for adversarial loss
    # lambda_adv = 0  # Weight for adversarial loss

    # lambda_adv = min(1.0, 0.1 * epoch)  # Increase over time
    loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    model.train()
    # private_encoder.train()
    # reconstruction_decoder.train()

    t1 = time.time()
    # print(train_loader)
    # print()

    # Cross-entropy loss for classification
    criterion_cls = nn.CrossEntropyLoss()


    for idx, (images, labels, metadata) in enumerate(train_loader):

        # print('THIS IS SHAPE OF IMAGES')
        # print(images[0].shape)
        # print(metadata)
        # [1,91,109,91]
        # print('hi')
        data_time.update(time.time() - t1)
        # print(images[0])



        
        # Ensure site_labels is a list of site names
        site_labels = list(metadata[1])  # Convert tuple to list if necessary
        # print("EXAMPLE site labvels:", site_labels[:10])
        site_labels = [int(label) for label in site_labels]

        # Convert site labels (strings) to numeric indices
        # label_encoder = LabelEncoder()
        # site_labels = label_encoder.fit_transform(site_labels)  # Converts strings to integers

        if opts.path == "local":
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

            # images = torch.cat(images, dim=0).to(opts.device)
            # print(images.shape)
            images = torch.cat(images, dim=0).to(device)
            # print(images.shape)

            bsz = labels.shape[0]
            labels = labels.float().to(device)

            # Convert to torch tensor
            site_labels = torch.tensor(site_labels, dtype=torch.long, device=device)
        else:
            # images = torch.cat(images, dim=0).to(opts.device)
            images = torch.cat(images, dim=0).to(opts.device)
            bsz = labels.shape[0]
            labels = labels.float().to(opts.device)

            # Convert to torch tensor
            site_labels = torch.tensor(site_labels, dtype=torch.long, device=opts.device)
        # print(site_labels)
        # site_labels = site_labels.repeat_interleave(opts.n_views)

        # if which_data_type == 'MREData':
            
        if opts.modality == "OpenBHB":
            site_labels = site_labels - 1

        # print('THIS IS SHAPE OF IMAGES BEFORE SQUEEZING')
        # print(images[0].shape)

        # CHANGED
        images = images.squeeze()
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ADDED THIS -=-==-=-=-=-=-=-=-=-=-=-=-=-=-==-
        # if which_data_type == 'MREData':




        if opts.modality == "OpenBHB":
            images = images.unsqueeze(1)  # Add channel dimension at index 1

        # print('THIS IS SHAPE OF IMAGES AFTER SQUEEZING')
        # print(images[0].shape)

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast(scaler is not None):
            # print("GNRGIRNGRIGNRIGRGNRGINGRIGNRGNRIGRIGNRGIRGRGNRGRIN")            
            # print(images.shape)


            # projected = model(images, classifier=True)

            # CHANGE BACK TO UNCOMMENTED

            images = images.contiguous()


            projected, site_pred = model(images, classify=True)


            print("Projected (mean, std):", projected.mean().item(), projected.std().item())


            # print(site_pred.shape)
            # Outputs: torch.Size([64, 6])
            

            # site_labels_MMD = site_labels.repeat(opts.n_views)

            # # CHANGE BACK 

            # # Could do class_loss here, but would need to double the site_labels shape (either doube like 1,1,2,2,3,3 or 1,2,3,1,2,3?)
            # site_labels = site_labels.repeat(opts.n_views)

            # if which_data_type == "OpenBHB":
            #     site_labels = site_labels - 1

            # class_loss = criterion_cls(site_pred, site_labels)

            # # END OF CHANGE BACK

            # if opts.confound_loss == "mmd":
            # else:
            #     mmd_loss = 0

            site_pred = torch.split(site_pred, [bsz]*opts.n_views, dim=0)
            projected = torch.split(projected, [bsz]*opts.n_views, dim=0)
            projected = torch.cat([f.unsqueeze(1) for f in projected], dim=1)
            site_pred = torch.cat([f.unsqueeze(1) for f in site_pred], dim=1)

            mmd_loss = 0
            coral_loss = 0
            hsic_loss = 0

            # Should probably not have this? LOOK ABOVE
            site_pred = site_pred.mean(dim=1) 
            projected_mmd = projected.mean(dim=1) 
            if opts.confound_loss == "mmd" or opts.confound_loss == "class+mmd":
                mmd_loss = mmd_calculator(opts, projected_mmd, site_labels)
            elif opts.confound_loss == "coral":
                coral_loss = coral_calculator(opts, projected_mmd, site_labels)
            elif opts.confound_loss == "hsic":
                hsic_loss = hsic_calculator(projected_mmd, site_labels)



            class_loss = criterion_cls(site_pred, site_labels)


            # print(site_pred.shape)
            # Outputs: torch.Size([32, 2, 6])

            if opts.loss_choice == "supcon" or opts.loss_choice == "RnC":
                if opts.path == "local":
                    running_loss = infonce(projected, labels.to(device).float())
                else:
                    running_loss = infonce(projected, labels.to(opts.device).float())

            elif opts.loss_choice == "dynamic":
                if opts.path == "local":
                    if opts.NN_nb_step_size > 0:
                        running_loss = infonce(features=projected,
                                            labels=labels.to(device),
                                            epoch=epoch)

                    else:
                        running_loss = infonce(features=projected,
                                            labels=labels.to(device))
                else:
                    if opts.NN_nb_step_size > 0:
                        running_loss = infonce(features=projected,
                                            labels=labels.to(opts.device),
                                            epoch=epoch)

                    else:
                        running_loss = infonce(features=projected,
                                            labels=labels.to(opts.device))


            # print(site_labels)
            # Compute classification loss
                        


            # OUTPUTTING ACCURACY
            # predicted_sites = site_pred.argmax(dim=1)  # Get predicted class indices
            # site_accuracy = (predicted_sites == site_labels).float().mean().item()  # Compute accuracy
            # print(f"Batch {idx}: Site Classifier Accuracy = {site_accuracy:.4f}")


            print("This is class loss:", class_loss)

            if opts.confound_loss == "classification":
                total_loss = running_loss - opts.lambda_adv * class_loss
            if opts.confound_loss == "classificationGRL":
                total_loss = running_loss + opts.lambda_adv * class_loss

                # total_loss = class_loss
            elif opts.confound_loss == "basic":
                total_loss =  running_loss
            elif opts.confound_loss == "mmd":
                total_loss = running_loss + opts.lambda_mmd * mmd_loss

            elif opts.confound_loss == "class+mmd":
                total_loss = running_loss + opts.lambda_mmd * mmd_loss - opts.lambda_adv * class_loss

            elif opts.confound_loss == "coral":
                total_loss = running_loss + opts.lambda_coral * coral_loss

            elif opts.confound_loss == "hsic":
                total_loss = running_loss + opts.lambda_hsic * hsic_loss


        optimizer.zero_grad()
        if scaler is None:
            total_loss.backward()
            if opts.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        else:
            scaler.scale(total_loss).backward()
            if opts.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()

        # for name, param in model.named_parameters():
        #     if "classifier" in name:
        #         print(f"{name} gradient norm: {param.grad.norm().item()}")
                
        loss.update(total_loss.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)


        print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                f"BT {batch_time.avg:.3f}\t"
                f"ETA {datetime.timedelta(seconds=eta)}\t"
                f"loss {loss.avg:.3f}\t")

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t")
    
    return loss.avg, class_loss, mmd_loss, coral_loss, hsic_loss, batch_time.avg, data_time.avg


def difference_loss(shared, private):
    shared = F.normalize(shared, dim=1)
    private = F.normalize(private, dim=1)
    return torch.mean((shared * private).sum(dim=1)**2)



def train_dsn(train_loader, model, infonce, optimizer, opts, epoch):
    # lambda_adv = 0.35  # Weight for adversarial loss
    # lambda_adv = 0  # Weight for adversarial loss

    # lambda_adv = min(1.0, 0.1 * epoch)  # Increase over time
    loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    model.train()
    # private_encoder.train()
    # reconstruction_decoder.train()

    t1 = time.time()
    # print(train_loader)
    # print()

    # Cross-entropy loss for classification
    criterion_cls = nn.CrossEntropyLoss()


    for idx, (images, labels, metadata) in enumerate(train_loader):

        # print('THIS IS SHAPE OF IMAGES')
        # print(images[0].shape)
        # print(metadata)
        # [1,91,109,91]
        # print('hi')
        data_time.update(time.time() - t1)
        # print(images[0])



        
        # Ensure site_labels is a list of site names
        site_labels = list(metadata[1])  # Convert tuple to list if necessary
        # print("EXAMPLE site labvels:", site_labels[:10])
        site_labels = [int(label) for label in site_labels]

        # Convert site labels (strings) to numeric indices
        # label_encoder = LabelEncoder()
        # site_labels = label_encoder.fit_transform(site_labels)  # Converts strings to integers

        if opts.path == "local":
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

            # images = torch.cat(images, dim=0).to(opts.device)
            # print(images.shape)
            images = torch.cat(images, dim=0).to(device)
            # print(images.shape)

            bsz = labels.shape[0]
            labels = labels.float().to(device)

            # Convert to torch tensor
            site_labels = torch.tensor(site_labels, dtype=torch.long, device=device)
        else:
            # images = torch.cat(images, dim=0).to(opts.device)
            images = torch.cat(images, dim=0).to(opts.device)
            bsz = labels.shape[0]
            labels = labels.float().to(opts.device)

            # Convert to torch tensor
            site_labels = torch.tensor(site_labels, dtype=torch.long, device=opts.device)
        # print(site_labels)
        # site_labels = site_labels.repeat_interleave(opts.n_views)

        # if which_data_type == 'MREData':
            
        if opts.modality == "OpenBHB":
            site_labels = site_labels - 1

        # print('THIS IS SHAPE OF IMAGES BEFORE SQUEEZING')
        # print(images[0].shape)

        # CHANGED
        images = images.squeeze()
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ADDED THIS -=-==-=-=-=-=-=-=-=-=-=-=-=-=-==-
        # if which_data_type == 'MREData':




        if opts.modality == "OpenBHB":
            images = images.unsqueeze(1)  # Add channel dimension at index 1

        # print('THIS IS SHAPE OF IMAGES AFTER SQUEEZING')
        # print(images[0].shape)

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast(scaler is not None):
            # print("GNRGIRNGRIGNRIGRGNRGINGRIGNRGNRIGRIGNRGIRGRGNRGRIN")            
            # print(images.shape)


            # projected = model(images, classifier=True)

            # CHANGE BACK TO UNCOMMENTED

            images = images.contiguous()

            shared_features, site_pred, private_features, recon = model(images, site_labels)


            # print("Projected (mean, std):", projected.mean().item(), projected.std().item())


            # print(site_pred.shape)
            # Outputs: torch.Size([64, 6])
            

            # site_labels_MMD = site_labels.repeat(opts.n_views)

            # # CHANGE BACK 

            # # Could do class_loss here, but would need to double the site_labels shape (either doube like 1,1,2,2,3,3 or 1,2,3,1,2,3?)
            # site_labels = site_labels.repeat(opts.n_views)

            # if which_data_type == "OpenBHB":
            #     site_labels = site_labels - 1

            # class_loss = criterion_cls(site_pred, site_labels)

            # # END OF CHANGE BACK

            # if opts.confound_loss == "mmd":
            # else:
            #     mmd_loss = 0

            site_pred = torch.split(site_pred, [bsz]*opts.n_views, dim=0)
            projected = torch.split(shared_features, [bsz]*opts.n_views, dim=0)
            projected = torch.cat([f.unsqueeze(1) for f in projected], dim=1)
            site_pred = torch.cat([f.unsqueeze(1) for f in site_pred], dim=1)

            mmd_loss = 0
            coral_loss = 0
            hsic_loss = 0
            # class_loss = 0

            # Should probably not have this? LOOK ABOVE
            site_pred = site_pred.mean(dim=1) 
            # projected_mmd = projected.mean(dim=1) 
            # if opts.confound_loss == "mmd" or opts.confound_loss == "class+mmd":
            #     mmd_loss = mmd_calculator(opts, projected_mmd, site_labels)
            # elif opts.confound_loss == "coral":
            #     coral_loss = coral_calculator(opts, projected_mmd, site_labels)
            # elif opts.confound_loss == "hsic":
            #     hsic_loss = hsic_calculator(projected_mmd, site_labels)



            class_loss = criterion_cls(site_pred, site_labels)



            # print(site_pred.shape)
            # Outputs: torch.Size([32, 2, 6])

            if opts.loss_choice == "supcon" or opts.loss_choice == "RnC":
                if opts.path == "local":
                    running_loss = infonce(projected, labels.to(device).float())
                else:
                    running_loss = infonce(projected, labels.to(opts.device).float())

            elif opts.loss_choice == "dynamic":
                if opts.path == "local":
                    if opts.NN_nb_step_size > 0:
                        running_loss = infonce(features=projected,
                                            labels=labels.to(device),
                                            epoch=epoch)

                    else:
                        running_loss = infonce(features=projected,
                                            labels=labels.to(device))
                else:
                    if opts.NN_nb_step_size > 0:
                        running_loss = infonce(features=projected,
                                            labels=labels.to(opts.device),
                                            epoch=epoch)

                    else:
                        running_loss = infonce(features=projected,
                                            labels=labels.to(opts.device))


            # print(site_labels)
            # Compute classification loss



            # OUTPUTTING ACCURACY
            # predicted_sites = site_pred.argmax(dim=1)  # Get predicted class indices
            # site_accuracy = (predicted_sites == site_labels).float().mean().item()  # Compute accuracy
            # print(f"Batch {idx}: Site Classifier Accuracy = {site_accuracy:.4f}")


            print("This is class loss:", class_loss)

            # # # Total loss = Contrastive Loss + Classification Loss
            # if opts.confound_loss == "classification":
            #     total_loss = running_loss - opts.lambda_adv * class_loss

            # if opts.confound_loss == "classificationGRL":
            #     total_loss = running_loss + opts.lambda_adv * class_loss
            #     # total_loss = class_loss
            # elif opts.confound_loss == "basic":
            #     total_loss =  running_loss
            # elif opts.confound_loss == "mmd":
            #     total_loss = running_loss + opts.lambda_mmd * mmd_loss

            # elif opts.confound_loss == "class+mmd":
            #     total_loss = running_loss + opts.lambda_mmd * mmd_loss - opts.lambda_adv * class_loss

            # elif opts.confound_loss == "coral":
            #     total_loss = running_loss + opts.lambda_coral * coral_loss

            # elif opts.confound_loss == "hsic":
            #     total_loss = running_loss + opts.lambda_hsic * hsic_loss
            
            recon_loss = F.mse_loss(recon, images)
            diff_loss = difference_loss(shared_features, private_features)

            total_loss = running_loss + opts.lambda_adv * class_loss + opts.lambda_recon * recon_loss + opts.lambda_diff * diff_loss

        optimizer.zero_grad()
        if scaler is None:
            total_loss.backward()
            if opts.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        else:
            scaler.scale(total_loss).backward()
            if opts.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()

        # for name, param in model.named_parameters():
        #     if "classifier" in name:
        #         print(f"{name} gradient norm: {param.grad.norm().item()}")
                
        loss.update(total_loss.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)


        print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                f"BT {batch_time.avg:.3f}\t"
                f"ETA {datetime.timedelta(seconds=eta)}\t"
                f"loss {loss.avg:.3f}\t")

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t")
    
    return loss.avg, class_loss, mmd_loss, coral_loss, hsic_loss, batch_time.avg, data_time.avg



def extract_features_for_umap(test_loader, model, opts, key, max_features=64):
    features_list = []
    labels_list = []
    metadata_list = []
    # print('THIS IS TBHE TOTAL NUMBER OF SAMPLES')
    # print(len(train_loader.dataset))

    model.eval()  # Set the model to evaluation modeS

    total_samples = 0  # Counter for how many samples we have processed

    with torch.no_grad():  # No need to track gradients during feature extraction
        for idx, (images, labels, metadata) in enumerate(test_loader):
            print(f"Processing batch {idx}, total samples: {total_samples}")

            # Move images and labels to the appropriate device
            if opts.path == "local":
                device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                # images = torch.cat(images, dim=0).to(device)
                labels = labels.float().to(device)
                images = images.to(device)
                # metadata = metadata.to(device)
            else:
                # images = torch.cat(images, dim=0).to(opts.device)
                labels = labels.float().to(opts.device)
                images = images.to(opts.device)

                # metadata = metadata.to(opts.device)


            # if which_data_type == 'MREData':
            images = images.squeeze()
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ADDED THIS -=-==-=-=-=-=-=-=-=-=-=-=-=-=-==-
            # if which_data_type == 'MREData':
            if opts.modality == "OpenBHB":
                images = images.unsqueeze(1)  # Add channel dimension at index 1

            # Extract features from the model
            images = images.contiguous()

            features = model.features(images)  # Get features from the model
            # print(metadata[key])


            if opts.path == "local":
                # Calculate how many samples to extract based on max_features
                remaining_samples = max_features - total_samples
                batch_size = features.size(0)
                # If the batch has more features than required, slice to get only the required number
                if batch_size > remaining_samples:
                    features = features[:remaining_samples]
                    labels = labels[:remaining_samples]
                    metadata = metadata[key][:remaining_samples]


            # Repeat the labels for each view (n_views=2)
            # repeated_labels = labels.unsqueeze(1).expand(-1, opts.n_views).contiguous().view(-1)
            # repeated_metadata = metadata.unsqueeze(1).expand(-1, opts.n_views).contiguous().view(-1)

            # Append features and labels to lists (convert to numpy for UMAP)
            features_list.append(features.cpu())  # Convert features to numpy
            labels_list.append(labels.cpu())  # Convert repeated labels to numpy
            metadata_list.append(metadata[key]) 




            total_samples += features.size(0)  # Update the total number of processed samples

            if opts.path == "local":
                # Stop processing if we've reached the desired number of samples
                if total_samples >= max_features:
                    break
            # print(total_samples)

        # Concatenate all features and labels into single arrays
        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        all_metadata = np.concatenate(metadata_list, axis=0)

    print(f"Extracted {len(all_features)} features and {len(all_labels)} labels")
    
    # Check if the number of features and labels match
    if len(all_features) != len(all_labels):
        raise ValueError(f"Mismatch: {len(all_features)} features vs {len(all_labels)} labels!")

    return all_features, all_labels, all_metadata







# Define the MMD function
def mmd_rbf(X, Y, gamma=1.0):
    XX = torch.exp(-gamma * torch.cdist(X, X, p=2).pow(2))
    YY = torch.exp(-gamma * torch.cdist(Y, Y, p=2).pow(2))
    XY = torch.exp(-gamma * torch.cdist(X, Y, p=2).pow(2))
    return XX.mean() + YY.mean() - 2 * XY.mean()







def visualise_umap(test_loader, model, opts, epoch=0):
    key = 1
    variable_of_interest = str(key)

    # Main script to run UMAP and plot the results
    features, labels, metadata = extract_features_for_umap(test_loader, model, opts, key)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Metadata shape: {metadata.shape}")

    if opts.modality == "OpenBHB":
        metadata = metadata - 1


    sil_score = silhouette_score(features, metadata)
    
    # Mutual Information
    mi_score = mutual_info_classif(features, metadata).mean()
    
    # Convert to tensors
    site_labels = torch.tensor(metadata, dtype=torch.long)
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

    # results = {
    #     "Silhouette Score": sil_score,
    #     "Mutual Information Score": mi_score,
    #     "MMD Scores": mmd_scores,
    #     "Mean MMD Score": mean_mmd_score
    # }

    wandb.log({'sil_score': sil_score})
    wandb.log({'mi_score': mi_score})
    wandb.log({'mmd_score': mean_mmd_score})




    # Perform UMAP dimensionality reduction
    umap_model = umap.UMAP(random_state=42)
    embedding_umap = umap_model.fit_transform(features)

    ages = labels  # Assuming 'labels' represents ages

    # Normalize the sizes based on age values for better scaling
    sizes = (ages - ages.min()) + 10  # Shift to avoid zero sizes
    sizes = (sizes / sizes.max()) * 100  # Normalize to range [10, 100]


    umap_df = pd.DataFrame(embedding_umap, columns=['UMAP 1', 'UMAP 2'])
    umap_df[variable_of_interest] = metadata

    col_pal_str = 'hsv'
    order = 1
    color_palette = sns.color_palette(col_pal_str, len(umap_df[variable_of_interest].unique()))[::order]
    print(umap_df[variable_of_interest].unique())


    # Create a scatter plot using Seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=umap_df,
        x='UMAP 1',
        y='UMAP 2',
        hue=variable_of_interest,
        palette=color_palette,
        size=sizes,  # Scale point size based on age
        sizes=(20, 200),  # Define size range for the points
        # hue='Label',  # Use labels for color coding (optional)
        alpha=0.5  # Transparency of points
    )


    # Customize plot appearance
    plt.title('UMAP of Feature Vectors', fontsize=16)
    plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside
    plt.tight_layout()

    if opts.path == "local":
        filename = f'umap_features_seaborn_plot_epoch_{epoch}_{opts.loss_choice}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save with high resolution
        print(f"UMAP plot saved to '{filename}'")
    else:
        filename = f'/home/afc53/images/umap_features_seaborn_plot_epoch_{epoch}_{opts.loss_choice}_{opts.modality}_{opts.confound_loss}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save with high resolution
        print(f"UMAP plot saved to '{filename}'")

    # Close the plot to free memory
    plt.close()

    if opts.path == "cluster":
        save_path = "/rds/user/afc53/hpc-work/saved_features/"

        # Save the original features, labels, and metadata
        np.save(os.path.join(save_path, f'features_before_reduction_epoch_{epoch}_{opts.loss_choice}_{opts.modality}_{opts.confound_loss}.npy'), features)
        np.save(os.path.join(save_path, f'labels_epoch_{epoch}_{opts.loss_choice}_{opts.modality}_{opts.confound_loss}.npy'), labels)
        np.save(os.path.join(save_path, f'metadata_epoch_{epoch}_{opts.loss_choice}_{opts.modality}_{opts.confound_loss}.npy'), metadata)

        # Save UMAP and t-SNE reduced features
        np.save(os.path.join(save_path, f'features_umap_epoch_{epoch}_{opts.loss_choice}_{opts.modality}_{opts.confound_loss}.npy'), embedding_umap)
        # np.save(os.path.join(save_path, f'features_tsne_epoch_{epoch}.npy'), embedding_tsne)



def training(seed=0):
    print('parsing arguments')



    start_time = time.time()  # Start the timer




    # # # FOR SWEEP
    # # if is_sweeping:
    # wandb.init(
    #     project="contrastive-brain-age-prediction",
    #     entity="afc53-university-of-cambridge"
    # )

    # wandb.init()

    opts = parse_arguments()

    print(opts.wandb_name)
    wandb.init(
    project=opts.wandb_name,
    entity="afc53-university-of-cambridge",
    # sync_tensorboard=True,
    settings=wandb.Settings(code_dir="/src"),
    tags=['to test'],
    # reinit=True,
    # config=opts
    )

    if is_sweeping:
        config = wandb.config
        # opts.trial = config.trial


        opts.weight_decay = config.weight_decay
        opts.noise_std = config.noise_std
        opts.lr = config.lr
        # opts.lambda_mmd = config.lambda_mmd
        opts.lambda_adv = config.lambda_adv
        opts.lambda_val = config.lambda_val
        


    print(opts.trial)

    # CHAnged from this to (seed)
    set_seed(opts.trial)
    # set_seed(seed)

    print('loading data')

    train_loader, train_loader_score, test_loader = load_data(opts)

    print('data loaded')
    if opts.path == "local":
        # Check if MPS is available
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    model, infonce = load_model(opts)
    optimizer = load_optimizer(model, opts)

    model_name = opts.model
    if opts.warm:
        model_name = f"{model_name}_warm"
    if opts.amp:
        model_name = f"{model_name}_amp"
    
    method_name = opts.method
    if opts.method == 'threshold':
        method_name = f"{method_name}_reduction_{opts.delta_reduction}"

    optimizer_name = opts.optimizer
    if opts.clip_grad:
        optimizer_name = f"{optimizer_name}_clipgrad"

    kernel_name = opts.kernel
    if opts.kernel == "gaussian" or opts.kernel == 'rbf':
        kernel_name = f"{kernel_name}_sigma{opts.sigma}"
    elif opts.kernel == 'cauchy':
        kernel_name = f"{kernel_name}_gamma{opts.sigma}"
    
    run_name = (f"{model_name}_{method_name}_"
                f"{optimizer_name}_"
                f"tf{opts.tf}_"
                f"lr{opts.lr}_{opts.lr_decay}_step{opts.lr_decay_step}_rate{opts.lr_decay_rate}_"
                f"temp{opts.temp}_"
                f"wd{opts.weight_decay}_"
                f"bsz{opts.batch_size}_views{opts.n_views}_"
                f"trainall_{opts.train_all}_"
                f"kernel_{kernel_name}_"
                f"method{opts.loss_choice}_"
                # f"f{opts.alpha}_lambd{opts.lambd}_"
                f"trial{opts.trial}")
    
    # tb_dir = os.path.join(opts.save_dir, "tensorboard", run_name)
    # save_dir = os.path.join(opts.save_dir, f"openbhb_models", run_name)
    # ensure_dir("scratch/output/brain-age-mri/tensorboard/experiment_001")
    # ensure_dir("scratch/output/brain-age-mri/openbhb_models/experiment_001")


    opts.model_class = model.__class__.__name__
    opts.criterion = infonce.__class__.__name__
    opts.optimizer_class = optimizer.__class__.__name__

    # wandb.init(project="brain-age-prediction", config=opts, name=run_name, sync_tensorboard=True,
    #           settings=wandb.Settings(code_dir="/src"), tags=['to test'])
    
    # COMMENTED OUT FOR SWEEP
    # if is_sweeping == False:
    # if wandb.sweep_id is None:
    # wandb.init(project='contrastive-brain-age-prediction', config=opts, name=run_name,
    #         settings=wandb.Settings(code_dir="/src"), tags=['to test'])


    print('Config:', opts)
    print('Model:', model.__class__.__name__)
    print('Criterion:', infonce)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    # writer = torch.utils.tensorboard.writer.SummaryWriter("scratch/output/brain-age-mri/tensorboard/experiment_001")
    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.

    # visualise_umap(test_loader, model, opts)


    # # Initialize the site classifier
    # # input_dim = model.projector.output_dim  # Adjust according to your model
    # num_sites = 6  # Number of unique sites (0-5)

    # if opts.path == "local":
    #     device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    #     site_classifier = SiteClassifier(128, num_sites).to(device)
    # else:
    #     site_classifier = SiteClassifier(128, num_sites).to(opts.device)

    # # Added
    # site_optimizer = torch.optim.Adam(site_classifier.parameters(), lr=1e-3)

    # # Added
    # scheduler = lr_scheduler.StepLR(site_optimizer, step_size=5, gamma=0.5)


    for epoch in range(1, opts.epochs + 1):

        # if epoch == 2:
        #     visualise_umap(test_loader, model, opts, epoch)
        #     mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        #     # writer.add_scalar("train/mae", mae_train, epoch)
        #     # writer.add_scalar("test/mae_int", mae_int, epoch)
        #     # writer.add_scalar("test/mae_ext", mae_ext, epoch)
        #     print("Age MAE:", mae_train, mae_test)

        # if epoch == 3:
        #     visualise_umap(test_loader, model, opts, epoch)
        #     mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        #     # writer.add_scalar("train/mae", mae_train, epoch)
        #     # writer.add_scalar("test/mae_int", mae_int, epoch)
        #     # writer.add_scalar("test/mae_ext", mae_ext, epoch)
        #     print("Age MAE:", mae_train, mae_test)

        # if epoch == 4:
        #     visualise_umap(test_loader, model, opts, epoch)
        #     mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        #     # writer.add_scalar("train/mae", mae_train, epoch)
        #     # writer.add_scalar("test/mae_int", mae_int, epoch)
        #     # writer.add_scalar("test/mae_ext", mae_ext, epoch)
        #     print("Age MAE:", mae_train, mae_test)

        # if epoch == 5:
        #     visualise_umap(test_loader, model, opts, epoch)
        #     mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        #     # writer.add_scalar("train/mae", mae_train, epoch)
        #     # writer.add_scalar("test/mae_int", mae_int, epoch)
        #     # writer.add_scalar("test/mae_ext", mae_ext, epoch)
        #     print("Age MAE:", mae_train, mae_test)



        adjust_learning_rate(opts, optimizer, epoch)

        t1 = time.time()
        # Changed
        if opts.confound_loss == "dsn":
            loss_train, class_loss_train, mmd_loss_train, coral_loss_train, hsic_loss_train, batch_time, data_time = train_dsn(train_loader, model, infonce, optimizer, opts, epoch)
        else:
            loss_train, class_loss_train, mmd_loss_train, coral_loss_train, hsic_loss_train, batch_time, data_time = train(train_loader, model, infonce, optimizer, opts, epoch)
        t2 = time.time()
        wandb.log({"train/loss": loss_train, "lr": optimizer.param_groups[0]['lr'], "BT": batch_time, "DT": data_time,
            "epoch": epoch})
        
        wandb.log({"train/class_loss": class_loss_train, "lr": optimizer.param_groups[0]['lr'], "BT": batch_time, "DT": data_time,
            "epoch": epoch})
        
        wandb.log({"train/mmd_loss": mmd_loss_train, "lr": optimizer.param_groups[0]['lr'], "BT": batch_time, "DT": data_time,
            "epoch": epoch})
        
        wandb.log({"train/coral_loss": coral_loss_train, "lr": optimizer.param_groups[0]['lr'], "BT": batch_time, "DT": data_time,
            "epoch": epoch})
        
        wandb.log({"train/hsic_loss": hsic_loss_train, "lr": optimizer.param_groups[0]['lr'], "BT": batch_time, "DT": data_time,
            "epoch": epoch})
        
        
        
        # writer.add_scalar("train/loss", loss_train, epoch)

        # writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        # writer.add_scalar("BT", batch_time, epoch)
        # writer.add_scalar("DT", data_time, epoch)
        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} loss {loss_train:.4f}")
        mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        wandb.log({"train/mae": mae_train, "test/mae": mae_test, "epoch": epoch})


        ba_train, ba_test = compute_site_ba(model, train_loader_score, test_loader, opts)
        wandb.log({"train/ba": ba_train, "test/ba": ba_test, "epoch": epoch})

        # if epoch % 5 == 0:
        #     ba_train, ba_test = compute_site_ba(model, train_loader_score, test_loader, opts)
        #     print("Site BA:", ba_train, ba_test)

        # if epoch % opts.save_freq == 0:
        #     # WAS ALREADY COMMENTED OUT 
        #     # save_file = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
        #     # save_model(model, optimizer, opts, epoch, save_file)

        #     mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        #     # writer.add_scalar("train/mae", mae_train, epoch)
        #     # writer.add_scalar("test/mae_int", mae_int, epoch)
        #     # writer.add_scalar("test/mae_ext", mae_ext, epoch)
        #     print("Age MAE:", mae_train, mae_test)

        #     # ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader_score, test_loader_int, test_loader_ext, opts)
        #     # # writer.add_scalar("train/site_ba", ba_train, epoch)
        #     # # writer.add_scalar("test/ba_int", ba_int, epoch)
        #     # # writer.add_scalar("test/ba_ext", ba_ext, epoch)
        #     # print("Site BA:", ba_train, ba_int, ba_ext)

        #     # challenge_metric = ba_int**0.3 * mae_ext
        #     # writer.add_scalar("test/score", challenge_metric, epoch)
        #     # print("Challenge score", challenge_metric)
    
        # # save_file = os.path.join(save_dir, f"weights.pth")
        # # save_model(model, optimizer, opts, epoch, save_file)
            
        # # Added
        # # scheduler.step()
            
    
    # visualise_umap(test_loader, model, opts, opts.epochs)

    
    mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
    # writer.add_scalar("train/mae", mae_train, epoch)
    # writer.add_scalar("test/mae_int", mae_int, epoch)
    # writer.add_scalar("test/mae_ext", mae_ext, epoch)
    print("Age MAE:", mae_train, mae_test)

    wandb.log({"train/mae": mae_train, "test/mae": mae_test, "epoch": opts.epochs})
    wandb.log({'mae_train': mae_train})
    wandb.log({'mae_test': mae_test})

    ba_train, ba_test = compute_site_ba(model, train_loader_score, test_loader, opts)

    wandb.log({"train/ba": ba_train, "test/ba": ba_test, "epoch": opts.epochs})
    wandb.log({'ba_train': ba_train})
    wandb.log({'ba_test': ba_test})
    # writer.add_scalar("train/site_ba", ba_train, epoch)
    # writer.add_scalar("test/ba_int", ba_int, epoch)
    # writer.add_scalar("test/ba_ext", ba_ext, epoch)
    print("Site BA:", ba_train, ba_test)
    
    # challenge_metric = ba_int**0.3 * mae_ext
    # writer.add_scalar("test/score", challenge_metric, epoch)
    # print("Challenge score", challenge_metric)

    if is_sweeping == False:
        visualise_umap(test_loader, model, opts)


    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print('TOTAL TIME TAKEN: ', elapsed_time)

    # # FOR SWEEP
    # if is_sweeping:
    #     wandb.finish()  # Close the WandB run
    # else:
    #     return mae_test, ba_test
    # if wandb.sweep_id is not None:
        # If it's a sweep (WandB sweep), end the current sweep run
    wandb.finish()






if __name__ == '__main__':

    if is_sweeping == True:
        opts = parse_arguments()
        sweep_config = {
            'method': 'bayes',
            # "name": "classification_tuning_dynamic_negative_classloss_noGRL_part2",
            # "name": "tuning_of_mmd_RnC_OpenBHB_1.0",
            # "name": f"tuning_of_basic_dynamic_OpenBHB",
            "name": f"part3_tuning_of_{opts.confound_loss}_{opts.loss_choice}_{opts.modality}",
            'metric': {
                'name': 'train/mae', #'mae_train'
                'goal': 'minimize'
            },
            "parameters": {

            "lr": {"values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]},
            # "weight_decay": {"values": [1e-6, 1e-2, 1e-4, 1e-5, 1e-3]},
            "weight_decay": {"values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]},
            "noise_std": {"values": [0, 0.01, 0.05, 0.1]},
                

            # Loss terms:
            "lambda_adv": {"values": [1e-2, 5e-2, 1e-1, 5e-1, 1]},
            "lambda_val": {"values": [1e-2, 5e-2, 1e-1, 5e-1, 1]},
            # "lambda_mmd": {"values": [1e-2, 5e-2, 1e-1, 5e-1, 1]},

        },
        }

        # args = parse_args()


        # sweep_id = wandb.sweep(sweep_config,
        #                        entity='jakobwandb',
        #                        #project='seedsNEW-pretrained-expw-finetune-sweeps-' + args.sweep + '-' + args.modality)
        #                        project='MICCAI_suppl')
        

        sweep_id = wandb.sweep(sweep_config, project="contrastive-brain-age-prediction")

        print(sweep_id)

        wandb.agent(sweep_id, function=training, count=20)

    else:
        training()





# mae_scores = []
# ba_scores = []
# for i in range(5):
#     mae_test, ba_test = training(seed=i)
#     mae_scores.append(mae_test)
#     ba_scores.append(ba_test)
#     wandb.log({"final_scores/ba": ba_test, "final_scores/mae": mae_test, "epoch": i})
# mae_mean = np.mean(mae_scores)
# ba_mean = np.mean(ba_scores)
# print("MAE MEAN:", mae_mean)
# print("BA MEAN:", ba_mean)

# wandb.log({"mean_score/ba": ba_mean, "mean_score/mae": mae_mean})


        

        

            
