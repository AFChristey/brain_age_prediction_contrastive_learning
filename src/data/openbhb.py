import numpy as np
import os
import nibabel
import torch
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import OrderedDict
# A utility for processing masked brain imaging data
from nilearn.masking import unmask
import re
import sys


import matplotlib.pyplot as plt
import seaborn as sns


# takes a tensor of real ages and bins them into age categories
# output is a tensor with binned age values
def bin_age(age_real: torch.Tensor):
    bins = [i for i in range(4, 92, 2)]
    age_binned = age_real.clone()
    for value in bins[::-1]:
        age_binned[age_real <= value] = value
    return age_binned.long()



def read_data(path, dataset):
    print(f"Read {dataset.upper()}")
    df = pd.read_csv(os.path.join(path + '/' + dataset + '_labels/', dataset + "_participants.tsv"), sep="\t")
    df['participant_id'] = df['participant_id'].astype(str)
 
 
    if dataset == "train":
        x_arr = np.load(os.path.join(path + '/' + dataset + '_quasiraw/' + dataset + '_quasiraw/',
                                     dataset + "_quasiraw_2mm.npy"), mmap_mode="r")
        participants_id = np.load(os.path.join(path + '/' + dataset + '_quasiraw/' + dataset + '_quasiraw/',
                                               "participants_id.npy"))
        # x_arr = x_arr[:1]
        # x_arr = x_arr[:300]

    elif dataset == "val":
        x_arr = np.load(os.path.join(path + '/' + dataset + '_quasiraw/', dataset + "_quasiraw_2mm.npy"), mmap_mode="r")
        participants_id = np.load(os.path.join(path + '/' + dataset + '_quasiraw/', "participants_id.npy"))
        # x_arr = x_arr[:1]
        # x_arr = x_arr[300:500]

    else:
        raise ValueError("Invalid dataset")
 
    matching_ages = df[df['participant_id'].isin(participants_id)][['participant_id', 'age', 'site', 'sex', 'study']]
    y_arr = matching_ages[['age', 'site', 'sex', 'study']].values

    # if dataset == "train":
    #     y_arr = y_arr[:1]
    #     # y_arr = y_arr[:300]
    # if dataset == "val":
    #     y_arr = y_arr[:1]
    #     # y_arr = y_arr[300:500]
    

    print("- y size [original]:", y_arr.shape)
    print("- x size [original]:", x_arr.shape)
    assert y_arr.shape[0] == x_arr.shape[0]

    if dataset == "train":
        # Convert y_arr to a DataFrame for easier manipulation
        df_plot = pd.DataFrame(y_arr, columns=['age', 'site', 'sex', 'study'])

        # Ensure 'age' is integer for proper binning
        df_plot['age'] = df_plot['age'].astype(int)

        # Count number of subjects per age per site
        age_site_counts = df_plot.groupby(['age', 'study']).size().unstack(fill_value=0)

        # Plot settings
        # plt.figure(figsize=(24, 15))
        fig, ax = plt.subplots(figsize=(12, 5))  # Increase width

        age_site_counts.plot(kind='bar', stacked=True, colormap='tab10', width=1.0, ax=ax, edgecolor="black", linewidth=1.2)


        # age_site_counts.plot(kind='bar', stacked=True, colormap='tab10', width=1.2)

        # Labels and title
        plt.xlabel("Age (Years)", fontsize=14)
        plt.ylabel("Number of Subjects", fontsize=14)
        plt.title("Number of Subjects vs. Age (Grouped by Study)", fontsize=16)
        plt.legend(title="Study", bbox_to_anchor=(1.05, 1), loc='upper left')

        # # Set x-axis ticks to display only every 10 years at the end of the bars
        # tick_positions = np.arange(0, len(age_site_counts), 10) + 0.5  # Shift by 0.5 to move to the right
        # tick_labels = age_site_counts.index[np.arange(0, len(age_site_counts), 10)]  # Corresponding age values
        # ax.set_xticks(tick_positions)
        # ax.set_xticklabels(tick_labels, rotation=0)  # Upright labels



        # Ensure x-axis ticks are at fixed intervals (0, 10, 20, ..., 90)
        min_age = 0  # Start at 0 (or you can use df_plot['age'].min())
        max_age = 90  # End at 90 (or you can use df_plot['age'].max())
        tick_positions = np.arange(min_age, max_age + 1, 10) + 0.5  # Shift by 0.5 to align with the right edge
        tick_positions = tick_positions[tick_positions <= age_site_counts.shape[0]]  # Ensure valid index range

        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(tick - 0.5)) for tick in tick_positions], rotation=0)  # Convert to integers



        # Save the plot
        plt.tight_layout()
        plt.savefig("/home/afc53/images/age_distribution_by_study.png", dpi=300)


    sys.exit()

        
    return x_arr, y_arr
 
 
class OpenBHB(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, path="local"):
        # self.root = root

        if path == "local":
            root = "data/results/openBHB"
        else:
            root = "/rds/user/afc53/hpc-work/MRE_Data/openBHB"

        self.train = train
        
        dataset = "train" if train else "val"
        
        self.X, self.y = read_data(root + '/' + dataset, dataset)
        self.T = transform
        
        print(f"Read {len(self.X)} records")
 
    def __len__(self):
        return len(self.y)
    

    def norm(self):

        default_value = 0
        self.X = norm_whole_batch(self.X, 'mean_std', default_value)

    def __getitem__(self, index):
 
        x = self.X[index]
        y = self.y[index]
 
        if self.T is not None:
            x = self.T(x)
        
        # sample, age, site
        age, site, sex = y[0], y[1], y[2]
 
        return x, age, (sex, site)





# OpenBHB is a custom PyTorch Dataset class that loads MRI data and labels (age, site) for training
class MREData(torch.utils.data.Dataset):
    # Depending on the train and internal flags, it loads the appropriate dataset (train, internal_test, or external_test)
    # root = root directory where the data is stored
    # label = specifies whether the labels should be continuous ("cont") or binned ("bin"). Defaults to "cont"
    # load_feats = If provided, it specifies a file to load additional biased features
    def __init__(self, modality, train=True, transform=None, 
                 label="cont", fast=False, load_feats=None, path="local", fold=0):
        # Stores the root path where the data is located as an instance variable self.root. 
        # This will be used to locate the files later


        (stiffness, dr, T1, age, sex, study,
         id, imbalance_percentages) = load_samples(path=path)

        sex = pd.Series(sex)
        sex = sex.replace('f', 'F')
        sex = sex.replace('m', 'M')
        sex = sex.to_numpy()

        if modality == 'stiffness':
            _, mu_stiff, sigma_stiff = normalize_mean_0_std_1(stiffness, default_value=0, mu_nonzero=None,
                                                              sigma_nonzero=None)
            [self.mu, self.sigma] = mu_stiff, sigma_stiff

        elif modality == 'dr':
            _, mu_dr, sigma_dr = normalize_mean_0_std_1(dr, default_value=0, mu_nonzero=None, sigma_nonzero=None)
            [self.mu, self.sigma] = mu_dr, sigma_dr

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        assert fold in range(5) or fold is None

        if fold in range(5):
            for fold_iter, (train_ids, test_ids) in enumerate(kfold.split(stiffness)):

                if fold_iter == fold:
                    stiffness_train, stiffness_test = stiffness[train_ids], stiffness[test_ids]
                    dr_train, dr_test = dr[train_ids], dr[test_ids]
                    T1_train, T1_test = T1[train_ids], T1[test_ids]
                    sex_train, sex_test = sex[train_ids], sex[test_ids]
                    age_train, age_test = age[train_ids], age[test_ids]
                    study_train, study_test = study[train_ids], study[test_ids]
                    imbalance_train, imbalance_test = imbalance_percentages[train_ids], imbalance_percentages[test_ids]
                    # MRE_coverage_train, MRE_coverage_test = MRE_coverage[train_ids], MRE_coverage[test_ids]

                else:
                    continue
        else:
            stiffness_train, stiffness_test, \
                dr_train, dr_test, \
                T1_train, T1_test, \
                age_train, age_test, \
                sex_train, sex_test, \
                study_train, study_test, \
                imbalance_train, imbalance_test = train_test_split(stiffness, dr, T1, age, sex, study,
                                                                         imbalance_percentages,
                                                                         test_size=0.2,
                                                                         random_state=42)

        # THIS IS TO TRAIN-TEST SPLIT (ONLY 60 TEST SAMPLES)
            
        if train:
            self.y = age_train
            self.sex = sex_train
            self.site = study_train
            self.imbalance = imbalance_train
            # self.MRE_coverage = MRE_coverage_train

            if modality == 'stiffness':
                self.x = stiffness_train
            elif modality == 'dr':
                self.x = dr_train
            elif modality == 'T1':
                self.x = T1_train

        else:
            self.y = age_test
            self.sex = sex_test
            self.site = study_test
            self.imbalance = imbalance_test
            # self.MRE_coverage = MRE_coverage_test

            if modality == 'stiffness':
                self.x = stiffness_test
            elif modality == 'dr':
                self.x = dr_test
            elif modality == 'T1':
                self.x = T1_test


        # THIS IS TO HAVE SAME TRAIN AND TEST - risky as overfitting
            
        # if train:
        #     self.y = age
        #     self.sex = sex
        #     self.site = study
        #     self.imbalance = imbalance_percentages
        #     # self.MRE_coverage = MRE_coverage_train

        #     if modality == 'stiffness':
        #         self.x = stiffness
        #     elif modality == 'dr':
        #         self.x = dr
        #     elif modality == 'T1':
        #         self.x = T1

        # else:
        #     self.y = age
        #     self.sex = sex
        #     self.site = study
        #     self.imbalance = imbalance_percentages
        #     # self.MRE_coverage = MRE_coverage_test

        #     if modality == 'stiffness':
        #         self.x = stiffness
        #     elif modality == 'dr':
        #         self.x = dr
        #     elif modality == 'T1':
        #         self.x = T1



        self.modality = modality
        self.T = transform

    def norm(self):

        default_value = 0

        if self.modality == 'T1':
            self.x = norm_whole_batch(self.x, 'mean_std', default_value)

        elif self.modality == 'dr' or self.modality == 'stiffness':
            self.x, _, _ = normalize_mean_0_std_1(self.x,
                                                  default_value=default_value,
                                                  mu_nonzero=self.mu,
                                                  sigma_nonzero=self.sigma)

        else:
            raise ValueError('Invalid modality')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = self.x[index]
        y = self.y[index]

        sex = self.sex[index]
        site = self.site[index]
        imbalance = self.imbalance[index]
        # MRE_coverage = self.MRE_coverage[index]

        if self.T is not None:
            x = self.T(x)
        else:
            x = torch.from_numpy(x).float()

        return x, y, (sex, site, imbalance)



def norm_whole_batch(batch, norm, default_value):
    batch_normed = np.zeros_like(batch)

    for i in range(batch.shape[0]):
        if norm == 'mean_std':
            batch_normed[i], _, _ = normalize_mean_0_std_1(batch[i], default_value, None, None)

        else:
            raise ValueError('norm has to be min_max or mean_std')

    return batch_normed


def normalize_mean_0_std_1(arr, default_value, mu_nonzero, sigma_nonzero):
    arr_nonzero = arr[np.nonzero(arr)]

    if mu_nonzero is None and sigma_nonzero is None:
        mu_nonzero = np.mean(arr_nonzero)
        sigma_nonzero = np.std(arr_nonzero)

    if default_value == 0:
        arr_pp = np.zeros_like(arr)

    elif default_value == -1:
        arr_pp = np.ones_like(arr) * -1

    else:
        raise ValueError('default_value has to be 0 or -1')

    arr_pp[np.nonzero(arr)] = (arr[np.nonzero(arr)] - mu_nonzero) / sigma_nonzero

    return arr_pp, mu_nonzero, sigma_nonzero



def load_samples(path):

    if path == 'local':
        prefix_path = 'data/results/Studies_healthy'
    else:
        prefix_path = '/home/afc53/contrastive_learning_mri_images/src/data/results/Studies_healthy'


    stiffness_ATLAS = np.load(prefix_path + '/ATLAS/stiffness_134.npy', allow_pickle=True)
    dr_ATLAS = np.load(prefix_path + '/ATLAS/dr_134.npy', allow_pickle=True)
    T1_ATLAS = np.load(prefix_path + '/ATLAS/T1_masked_ATLAS_cluster.npy', allow_pickle=True)  # T1_ATLAS.npy
    age_ATLAS = np.load(prefix_path + '/ATLAS/age_ATLAS.npy', allow_pickle=True)
    sex_ATLAS = np.load(prefix_path + '/ATLAS/sex_ATLAS.npy', allow_pickle=True)
    id_ATLAS = np.load(prefix_path + '/ATLAS/id_ATLAS.npy', allow_pickle=True)
    # MRE_coverage_ATLAS = np.load(prefix_path + '/ATLAS/MRE_coverage_ATLAS.npy', allow_pickle=True)
    study_ATLAS = np.array([0] * len(age_ATLAS))

    # stiffness_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/stiffness_OA.npy', allow_pickle=True)
    # dr_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/dr_OA.npy', allow_pickle=True)
    # T1_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/T1_masked_OA.npy', allow_pickle=True)  # T1_OA.npy
    # age_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/age_OA.npy', allow_pickle=True)
    # sex_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/sex_OA.npy', allow_pickle=True)
    # id_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/id_OA.npy', allow_pickle=True)
    # MRE_coverage_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/MRE_coverage_OA.npy', allow_pickle=True)
    # study_OA = np.array(['CN'] * len(age_OA))

    stiffness_BMI = np.load(prefix_path + '/BMI/stiffness_BMI.npy', allow_pickle=True)
    dr_BMI = np.load(prefix_path + '/BMI/dr_BMI.npy', allow_pickle=True)
    T1_BMI = np.load(prefix_path + '/BMI/T1_masked_BMI.npy', allow_pickle=True)  # T1_MIMS.npy
    age_BMI = np.load(prefix_path + '/BMI/age_BMI.npy', allow_pickle=True)
    sex_BMI = np.load(prefix_path + '/BMI/sex_BMI.npy', allow_pickle=True)
    id_BMI = np.load(prefix_path + '/BMI/id_BMI.npy', allow_pickle=True)
    # MRE_coverage_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/MRE_coverage_BMI.npy', allow_pickle=True)
    study_BMI = np.array([1] * len(age_BMI))

    stiffness_NOVA = np.load(prefix_path + '/NOVA/stiffness_NOVA.npy', allow_pickle=True)
    dr_NOVA = np.load(prefix_path + '/NOVA/dr_NOVA.npy', allow_pickle=True)
    T1_NOVA = np.load(prefix_path + '/NOVA/T1_masked_NOVA.npy', allow_pickle=True)  # T1_MIMS.npy
    age_NOVA = np.load(prefix_path + '/NOVA/age_NOVA.npy', allow_pickle=True)
    sex_NOVA = np.load(prefix_path + '/NOVA/sex_NOVA.npy', allow_pickle=True)
    id_NOVA = np.load(prefix_path + '/NOVA/id_NOVA.npy', allow_pickle=True)
    # MRE_coverage_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/MRE_coverage_NOVA.npy', allow_pickle=True)
    study_NOVA = np.array([2] * len(age_NOVA))

    stiffness_NITRC_batch_1 = np.load(prefix_path + '/NITRC_batch_1/stiffness_NITRC_batch_1.npy',
                                      allow_pickle=True)
    dr_NITRC_batch_1 = np.load(prefix_path + '/NITRC_batch_1/dr_NITRC_batch_1.npy', allow_pickle=True)
    T1_NITRC_batch_1 = np.load(prefix_path + '/NITRC_batch_1/T1_masked_NITRC_batch_1.npy',
                               allow_pickle=True)  # T1_NITRC_batch_1.npy
    age_NITRC_batch_1 = np.load(prefix_path + '/NITRC_batch_1/age_NITRC_batch_1.npy', allow_pickle=True)
    sex_NITRC_batch_1 = np.load(prefix_path + '/NITRC_batch_1/sex_NITRC_batch_1.npy', allow_pickle=True)
    id_NITRC_batch_1 = np.load(prefix_path + '/NITRC_batch_1/id_NITRC_batch_1.npy', allow_pickle=True)
    # MRE_coverage_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/MRE_coverage_NITRC_batch_1.npy',
    #                                      allow_pickle=True)
    study_NITRC_batch_1 = np.array([3] * len(age_NITRC_batch_1))

    stiffness_NITRC_batch_2 = np.load(prefix_path + '/NITRC_batch_2/stiffness_NITRC_batch_2.npy',
                                      allow_pickle=True)
    dr_NITRC_batch_2 = np.load(prefix_path + '/NITRC_batch_2/dr_NITRC_batch_2.npy', allow_pickle=True)
    T1_NITRC_batch_2 = np.load(prefix_path + '/NITRC_batch_2/T1_masked_NITRC_batch_2.npy',
                               allow_pickle=True)  # T1_NITRC_batch_2.npy
    age_NITRC_batch_2 = np.load(prefix_path + '/NITRC_batch_2/age_NITRC_batch_2.npy', allow_pickle=True)
    sex_NITRC_batch_2 = np.load(prefix_path + '/NITRC_batch_2/sex_NITRC_batch_2.npy', allow_pickle=True)
    id_NITRC_batch_2 = np.load(prefix_path + '/NITRC_batch_2/id_NITRC_batch_2.npy', allow_pickle=True)
    # MRE_coverage_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/MRE_coverage_NITRC_batch_2.npy',
    #                                      allow_pickle=True)
    study_NITRC_batch_2 = np.array([4] * len(age_NITRC_batch_2))

    stiffness_MIMS = np.load(prefix_path + '/MIMS/stiffness_MIMS.npy', allow_pickle=True)
    dr_MIMS = np.load(prefix_path + '/MIMS/dr_MIMS.npy', allow_pickle=True)
    T1_MIMS = np.load(prefix_path + '/MIMS/T1_masked_MIMS.npy', allow_pickle=True)  # T1_MIMS.npy
    age_MIMS = np.load(prefix_path + '/MIMS/age_MIMS.npy', allow_pickle=True)
    sex_MIMS = np.load(prefix_path + '/MIMS/sex_MIMS.npy', allow_pickle=True)
    id_MIMS = np.load(prefix_path + '/MIMS/id_MIMS.npy', allow_pickle=True)
    # MRE_coverage_MIMS = np.load(prefix_path + '/MIMS/MRE_coverage_MIMS.npy', allow_pickle=True)
    study_MIMS = np.array([5] * len(age_MIMS))


    stiffness_all_healthy = np.concatenate((stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_MIMS, stiffness_BMI, stiffness_NOVA), axis=0)
    dr_all_healthy = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_MIMS, dr_BMI, dr_NOVA),
                                    axis=0)
    T1_all_healthy = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_MIMS, T1_BMI, T1_NOVA),
                                    axis=0)
    age_all_healthy = np.concatenate(
        (age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_MIMS, age_BMI, age_NOVA), axis=0)
    sex_all_healthy = np.concatenate(
        (sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_MIMS, sex_BMI, sex_NOVA), axis=0)
    study_all_healthy = np.concatenate(
        (study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_MIMS, study_BMI, study_NOVA),
        axis=0)
    id_all_healthy = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_MIMS, id_BMI, id_NOVA),
                                    axis=0)
    # MRE_coverage_all_healthy = np.concatenate((MRE_coverage_ATLAS, MRE_coverage_NITRC_batch_1,
    #                                             MRE_coverage_NITRC_batch_2, MRE_coverage_OA, MRE_coverage_MIMS,
    #                                             MRE_coverage_BMI, MRE_coverage_NOVA), axis=0)

    unique, inverse = np.unique(age_all_healthy, return_inverse=True)
    counts = np.bincount(inverse)
    total_count = age_all_healthy.shape[0]
    imbalance_percentages = counts[inverse] / total_count

    return (
        stiffness_all_healthy, dr_all_healthy, T1_all_healthy, age_all_healthy, sex_all_healthy, study_all_healthy,
        id_all_healthy, imbalance_percentages)



class FeatureExtractor(BaseEstimator, TransformerMixin):
    """ Select only the requested data associatedd features from the the
    input buffered data.
    """
    print('getting modalities')
    MODALITIES = OrderedDict([
        ("vbm", {
            "shape": (1, 121, 145, 121),
            "size": 519945}),
        ("quasiraw", {
            "shape": (1, 182, 218, 182),
            "size": 1827095}),
        ("xhemi", {
            "shape": (8, 163842),
            "size": 1310736}),
        ("vbm_roi", {
            "shape": (1, 284),
            "size": 284}),
        ("desikan_roi", {
            "shape": (7, 68),
            "size": 476}),
        ("destrieux_roi", {
            "shape": (7, 148),
            "size": 1036})
    ])
    MASKS = {
        "vbm": {
            "path": None,
            "thr": 0.05},
        "quasiraw": {
            "path": None,
            "thr": 0}
    }

    def __init__(self, dtype, path, mock=False):
        """ Init class.
        Parameters
        ----------
        dtype: str
            the requested data: 'vbm', 'quasiraw', 'vbm_roi', 'desikan_roi',
            'destrieux_roi' or 'xhemi'.
        """
        print('init modalities')
        if dtype not in self.MODALITIES:
            raise ValueError("Invalid input data type.")
        self.dtype = dtype

        data_types = list(self.MODALITIES.keys())
        index = data_types.index(dtype)
        
        cumsum = np.cumsum([item["size"] for item in self.MODALITIES.values()])
        
        if index > 0:
            self.start = cumsum[index - 1]
        else:
            self.start = 0
        self.stop = cumsum[index]
        # print(self.stop)
        # print(self.start)
        
        self.masks = dict((key, val["path"]) for key, val in self.MASKS.items())
        if path == "local":
            self.masks["vbm"] = "data/masks/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
            self.masks["quasiraw"] = "data/masks/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"
        else:
            self.masks["vbm"] = "/home/afc53/contrastive_learning_mri_images/src/data/masks/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
            self.masks["quasiraw"] = "/home/afc53/contrastive_learning_mri_images/src/data/masks/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"

        self.mock = mock
        if mock:
            return

        for key in self.masks:
            if self.masks[key] is None or not os.path.isfile(self.masks[key]):
                raise ValueError("Impossible to find mask:", key, self.masks[key])
            arr = nibabel.load(self.masks[key]).get_fdata()
            thr = self.MASKS[key]["thr"]
            arr[arr <= thr] = 0
            arr[arr > thr] = 1
            # self.masks[key] = nibabel.Nifti1Image(arr.astype(int), np.eye(4))
            self.masks[key] = nibabel.Nifti1Image(arr.astype(np.int32), np.eye(4))


    def fit(self, X, y):
        return self

    def transform(self, X):
        # print(X)
        if self.mock:
            # print("transforming", X.shape)
            data = X.reshape(self.MODALITIES[self.dtype]["shape"])
            # print("mock data:", data.shape)
            return data
        
        X_flat = X.flatten()
        # print("Flattened X shape:", X_flat.shape)  # Debugging step

        select_X = X_flat[self.start:self.stop]

        
        # print(X.shape)
        # select_X = X[self.start:self.stop]
        if self.dtype in ("vbm", "quasiraw"):
            print("Shape of select_X before unmask:", select_X.shape)
            print(f"select_X shape: {select_X.shape}, expected features: {self.masks[self.dtype].shape}")
            im = unmask(select_X, self.masks[self.dtype])
            select_X = im.get_fdata()
            select_X = select_X.transpose(2, 0, 1)
        select_X = select_X.reshape(self.MODALITIES[self.dtype]["shape"])
        # print('transformed.shape', select_X.shape)
        return select_X
