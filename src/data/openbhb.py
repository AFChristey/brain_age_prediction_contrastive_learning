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


# takes a tensor of real ages and bins them into age categories
# output is a tensor with binned age values
def bin_age(age_real: torch.Tensor):
    bins = [i for i in range(4, 92, 2)]
    age_binned = age_real.clone()
    for value in bins[::-1]:
        age_binned[age_real <= value] = value
    return age_binned.long()





# OpenBHB is a custom PyTorch Dataset class that loads MRI data and labels (age, site) for training
class OpenBHB(torch.utils.data.Dataset):
    # Depending on the train and internal flags, it loads the appropriate dataset (train, internal_test, or external_test)
    # root = root directory where the data is stored
    # label = specifies whether the labels should be continuous ("cont") or binned ("bin"). Defaults to "cont"
    # load_feats = If provided, it specifies a file to load additional biased features
    def __init__(self, train=True, transform=None, 
                 label="cont", fast=False, load_feats=None, path="local", fold=0):
        # Stores the root path where the data is located as an instance variable self.root. 
        # This will be used to locate the files later


        (T1, age, study, sex) = load_samples_OpenBHB(path=path)


        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        assert fold in range(5) or fold is None

        if fold in range(5):
            for fold_iter, (train_ids, test_ids) in enumerate(kfold.split(T1)):

                if fold_iter == fold:
                    T1_train, T1_test = T1[train_ids], T1[test_ids]
                    sex_train, sex_test = sex[train_ids], sex[test_ids]
                    age_train, age_test = age[train_ids], age[test_ids]
                    study_train, study_test = study[train_ids], study[test_ids]
                    # MRE_coverage_train, MRE_coverage_test = MRE_coverage[train_ids], MRE_coverage[test_ids]

                else:
                    continue
        else:
            T1_train, T1_test, \
                age_train, age_test, \
                sex_train, sex_test, \
                study_train, study_test = train_test_split(T1, age, sex, study, test_size=0.2, random_state=42)

        if train:
            self.y = age_train
            self.sex = sex_train
            self.site = study_train
            # self.MRE_coverage = MRE_coverage_train
            self.x = T1_train

        else:
            self.y = age_test
            self.sex = sex_test
            self.site = study_test
            # self.MRE_coverage = MRE_coverage_test
            self.x = T1_test


        self.T = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = self.x[index]
        y = self.y[index]

        sex = self.sex[index]
        site = self.site[index]
        # MRE_coverage = self.MRE_coverage[index]

        if self.T is not None:
            x = self.T(x)
        else:
            x = torch.from_numpy(x).float()

        return x, y, (sex, site)



def load_samples_OpenBHB(path):

    if path == 'local':
        folder_path = 'data/results/OpenBHB_data/train_quasiraw'
        tsv_path = 'data/results/OpenBHB_data/participants.tsv'
    else:
        folder_path = '/rds/user/afc53/hpc-work/MRE_Data/OpenBHB_data/train_quasiraw'
        tsv_path = '/rds/user/afc53/hpc-work/MRE_Data/OpenBHB_data/participants.tsv'


    df_participants = pd.read_csv(tsv_path, sep="\t")

    # Ensure column names are correct
    participant_column = "participant_id"  # Adjust if needed
    age_column = "age"  # Change if the column name is different
    site_column = "study"  # Change if needed
    sex_column = "sex"  # Change if needed


    # Get list of all .npy files in the folder (without sorting)
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    print(npy_files[0])
    # Select the first 100 files (without sorting)

    if path == 'local':
        npy_files = npy_files[:100]

    # Initialize lists to store T1 data and metadata
    t1_data_list = []
    age_list = []
    study_list = []
    sex_list = []

    for file in npy_files:
        file_path = os.path.join(folder_path, file)

        match = re.search(r"sub-(\d+)_preproc-quasiraw_T1w.npy", file)
        if match:
            participant_id = match.group(1)  # Extract the ID as a string

            # 🔍 Find corresponding metadata in the TSV
            metadata_row = df_participants[df_participants[participant_column] == int(participant_id)]
            
            # Load T1 MRI Data
            data = np.load(file_path, allow_pickle=True)

            # Append MRI data
            t1_data_list.append(data)

            # Append metadata separately
            age_list.append(metadata_row.iloc[0][age_column])
            study_list.append(metadata_row.iloc[0][site_column])
            sex_list.append(metadata_row.iloc[0][sex_column])


    # ✅ Convert lists to structured NumPy formats
    t1_array = np.array(t1_data_list)  # NumPy array for MRI data
    age_array = np.array(age_list)
    study_array = np.array(study_list)
    sex_array = np.array(sex_list)

    # # 📌 Print shapes to verify
    # print("T1 Data Array Shape:", t1_array.shape)  # Expected: (100, 182, 218, 182) if each image is 3D
    # print("Age Array Shape:", age_array.shape)  # Expected: (100,)
    # print("Site Array Shape:", study_array.shape)  # Expected: (100,)
    # print("Sex Array Shape:", sex_array.shape)  # Expected: (100,)

    # print(t1_array[0])

    return (t1_array, age_array, study_array, sex_array)






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



# class FeatureExtractor(BaseEstimator, TransformerMixin):
#     """ Select only the requested data associatedd features from the the
#     input buffered data.
#     """
#     # OrderedDict where each key corresponds to a type of MRI data (or modality), 
#     # and the value contains information about its shape and size.
#     MODALITIES = OrderedDict([
#         ("vbm", {
#             "shape": (1, 121, 145, 121),
#             "size": 519945}),
#         ("quasiraw", {
#             "shape": (1, 182, 218, 182),
#             "size": 1827095}),
#         ("xhemi", {
#             "shape": (8, 163842),
#             "size": 1310736}),
#         ("vbm_roi", {
#             "shape": (1, 284),
#             "size": 284}),
#         ("desikan_roi", {
#             "shape": (7, 68),
#             "size": 476}),
#         ("destrieux_roi", {
#             "shape": (7, 148),
#             "size": 1036})
#     ])
#     # This dictionary defines the mask settings for each modality
#     # path = The path to the mask file (this can be None if no mask is needed).
#     # thr = The threshold for the mask. Values below this threshold will be masked out (set to 0).
#     MASKS = {
#         "vbm": {
#             "path": None,
#             "thr": 0.05},
#         "quasiraw": {
#             "path": None,
#             "thr": 0}
#     }

#     # initializes the FeatureExtractor object
#     # dtype = The data type (modality) to work with, e.g., "vbm", "quasiraw", etc.
#     # mock: A flag to simulate the transformation without actually loading the data (useful for debugging).
#     def __init__(self, dtype, mock=False):
#         """ Init class.
#         Parameters
#         ----------
#         dtype: str
#             the requested data: 'vbm', 'quasiraw', 'vbm_roi', 'desikan_roi',
#             'destrieux_roi' or 'xhemi'.
#         """
#         if dtype not in self.MODALITIES:
#             raise ValueError("Invalid input data type.")
#         self.dtype = dtype

#         data_types = list(self.MODALITIES.keys())
#         index = data_types.index(dtype)
        
#         # calculates the cumulative sum of sizes for each modality. 
#         # This helps determine where the current modality starts and stops in the input data
#         cumsum = np.cumsum([item["size"] for item in self.MODALITIES.values()])
        
#         # slice the data corresponding to the selected modality
#         if index > 0:
#             self.start = cumsum[index - 1]
#         else:
#             self.start = 0
#         self.stop = cumsum[index]
        
#         # creates a dictionary of masks with paths for each modality. For "vbm" and "quasiraw", paths to specific mask files are assigned
#         # mask files are used to filter the MRI data (i.e., zeroing out areas of the brain that are not of interest)
#         self.masks = dict((key, val["path"]) for key, val in self.MASKS.items())
#         self.masks["vbm"] = "./data/masks/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
#         self.masks["quasiraw"] = "./data/masks/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"

#         # If mock is True, the method exits early and skips further processing, 
#         # which means no actual loading or transformation of data occurs.
#         # (useful for debugging or testing)
#         self.mock = mock
#         if mock:
#             return

#         # iterates over the defined mask paths. For each mask:
#         for key in self.masks:
#             # It checks if the mask file exists.
#             if self.masks[key] is None or not os.path.isfile(self.masks[key]):
#                 raise ValueError("Impossible to find mask:", key, self.masks[key])
#             # Loads the mask using nibabel.load() and gets the mask data with .get_fdata().
#             arr = nibabel.load(self.masks[key]).get_fdata()
#             # Applies the threshold (thr): values below the threshold are set to 0, and values above the threshold are set to 1, essentially creating a binary mask.
#             thr = self.MASKS[key]["thr"]
#             arr[arr <= thr] = 0
#             arr[arr > thr] = 1
#             # self.masks[key] = nibabel.Nifti1Image(arr.astype(int), np.eye(4))
#             # The mask is then stored as a Nifti1Image object, which is used to apply the mask to the MRI data later.
#             self.masks[key] = nibabel.Nifti1Image(arr.astype(np.int32), np.eye(4))


#     def fit(self, X, y):
#         return self

#     # transform() method performs the actual transformation of the input data (X).
#     def transform(self, X):
#         # If mock is True, it reshapes the data to match the specified shape for the selected modality. 
#         # This is used for testing or simulation without applying real transformations.
#         if self.mock:
#             #print("transforming", X.shape)
#             data = X.reshape(self.MODALITIES[self.dtype]["shape"])
#             #print("mock data:", data.shape)
#             return data
        
#         # print(X.shape)
#         # slices the input data X according to the modality’s start and stop indices
#         select_X = X[self.start:self.stop]
#         # specific modalities ("vbm", "quasiraw"), it applies the mask using the unmask() function, 
#         # which masks out the irrelevant parts of the MRI data based on the binary mask.
#         if self.dtype in ("vbm", "quasiraw"):
#             im = unmask(select_X, self.masks[self.dtype])
#             # get_fdata() method of a Nifti1Image object retrieves the image data as a numpy array. 
#             # This data contains the intensity values of the MRI image, with the mask applied, 
#             # which means irrelevant or unimportant regions (those masked out) will have been set to 0 or removed.
#             select_X = im.get_fdata()
#             # data is transposed to ensure it matches the required shape (depth, height, width) for the MRI data
#             select_X = select_X.transpose(2, 0, 1)
#         # after unmasking and transposing, the data is reshaped to match the required shape for the selected modality.
#         select_X = select_X.reshape(self.MODALITIES[self.dtype]["shape"])
#         # print('transformed.shape', select_X.shape)
#         return select_X


# if __name__ == '__main__':
#     import sys
#     from torchvision import transforms
#     # Imports custom transformations Crop and Pad from a local module called transforms. 
#     # These transformations are defined in transforms.py
#     from .transforms import Crop, Pad

#     # This initializes an instance of the FeatureExtractor class with the modality "vbm". 
#     # The FeatureExtractor class is responsible for selecting and transforming the relevant features from the MRI data based on the modality (in this case, "vbm").
#     selector = FeatureExtractor("vbm")

#     # This method processes the input x (MRI data), likely extracting relevant features and transforming it according to the vbm modality.
#     T_pre = transforms.Lambda(lambda x: selector.transform(x))
#     # used to chain multiple transformations together into a single transformation pipeline
#     T_train = transforms.Compose([
#         T_pre,
#         Crop((1, 121, 128, 121), type="random"),
#         Pad((1, 128, 128, 128)),
#         transforms.Lambda(lambda x: torch.from_numpy(x)),
#         transforms.Normalize(mean=0.0, std=1.0)
#     ])

#     train_loader = torch.utils.data.DataLoader(OpenBHB(sys.argv[1], train=True, internal=True, transform=T_train),
#                                                batch_size=3, shuffle=True, num_workers=1,
#                                                persistent_workers=True)
    
#     # reates an iterator over the train_loader DataLoader object, and next() fetches the next batch of data from the iterator.
#     x, y1, y2 = next(iter(train_loader))
#     print(x.shape, y1, y2)