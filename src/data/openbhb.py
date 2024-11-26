import numpy as np
import os
import nibabel
import torch
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from collections import OrderedDict
# A utility for processing masked brain imaging data
from nilearn.masking import unmask

# takes a tensor of real ages and bins them into age categories
# output is a tensor with binned age values
def bin_age(age_real: torch.Tensor):
    bins = [i for i in range(4, 92, 2)]
    age_binned = age_real.clone()
    for value in bins[::-1]:
        age_binned[age_real <= value] = value
    return age_binned.long()

# reads both the imaging data (.npy file) and metadata (.tsv file)
# dataset specifies the dataset name (train, internal_test, or external_test)
def read_data(path, dataset, fast):
    # print(f"Read {dataset.upper()}")
    # The sep="\t" argument specifies that the values in the .tsv file are separated by tabs.
    # print(path)
    df = pd.read_csv("data/results/labels_with_sites.csv")
    # "site" column is set to NaN for rows in the "external_test" split, possibly because the "site" information is not available or relevant for external test data
    # df.loc[df["split"] == "external_test", "site"] = np.nan

    # creates a NumPy array (y_arr) containing the values of the "age" and "site" columns from the df DataFrame
    y_arr = df[["age", "site"]].values

    # initializes an empty NumPy array x_arr with the shape (10, 3659572)
    # 10 channels and 3,659,5732 features
    # x_arr = np.zeros((10, 3659572))
    # If fast is False, the actual MRI data (.npy file) is loaded into memory; 
    # otherwise, the array is initialized but not fully loaded into memory (mmap_mode="r")
    # if not fast:
    x_arr = np.load("data/results/x_arr.npy", mmap_mode="r")
    
    print("- y size [original]:", y_arr.shape)
    print("- x size [original]:", x_arr.shape)
    return x_arr, y_arr

# OpenBHB is a custom PyTorch Dataset class that loads MRI data and labels (age, site) for training
class OpenBHB(torch.utils.data.Dataset):
    # Depending on the train and internal flags, it loads the appropriate dataset (train, internal_test, or external_test)
    # root = root directory where the data is stored
    # label = specifies whether the labels should be continuous ("cont") or binned ("bin"). Defaults to "cont"
    # load_feats = If provided, it specifies a file to load additional biased features
    def __init__(self, root, train=True, internal=True, transform=None, 
                 label="cont", fast=False, load_feats=None):
        # Stores the root path where the data is located as an instance variable self.root. 
        # This will be used to locate the files later
        self.root = root

        # checks for an invalid configuration
        if train and not internal:
            raise ValueError("Invalid configuration train=True and internal=False")
        
        # store the train and internal flags as instance variables
        self.train = train
        self.internal = internal
        
        # This way, the class knows whether to load the training data, the internal test data, or the external test data
        dataset = "train"
        if not train:
            if internal:
                dataset = "internal_test"
            else:
                dataset = "external_test"
        
        # load the data from the disk.
        self.X, self.y = read_data(root, dataset, fast)
        # stores the transformation function (if provided) in self.T
        self.T = transform
        self.label = label
        self.fast = fast

        self.bias_feats = None
        if load_feats:
            print("Loading biased features", load_feats)
            # It loads the additional biased features from the specified file (load_feats) using torch.load
            self.bias_feats = torch.load(load_feats, map_location="cpu")
        
        # prints the number of records (data samples) in self.X, which is the feature set
        print(f"Read {len(self.X)} records")

    # len(self.y) is returned, which is the length of the labels array self.y. 
    # This corresponds to the number of samples in the dataset.
    def __len__(self):
        return len(self.y)

    # method defines how to access an individual item from the dataset (i.e., a single sample) given its index
    def __getitem__(self, index):
        # If fast is False, the feature data (self.X) is indexed using the provided index, which fetches the corresponding feature vector (MRI data) for that sample.
        # self.X[0]: If fast is True, instead of fetching a specific sample from the dataset, it always fetches the first sample (self.X[0])
        if not self.fast:
            x = self.X[index]
        else:
            x = self.X[0]

        # y will contain two values: age and site
        y = self.y[index]

        # Checks if a transformation (self.T) has been provided for the features.
        # If a transformation exists, it is applied to the feature data (x).
        if self.T is not None:
            x = self.T(x)
        
        # This unpacks the y array into two variables: age and site.
        age, site = y[0], y[1] 
        #  If so, it means the task is to classify age into bins instead of predicting the exact age.
        if self.label == "bin":
            age = bin_age(torch.tensor(age))
        
        # checks if any biased features (self.bias_feats) are available. 
        # If so, it returns the transformed feature data (x), the binned or continuous age, and the corresponding biased feature for the sample at the given index.
        if self.bias_feats is not None:
            return x, age, self.bias_feats[index]
        else:
            return x, age, site

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """ Select only the requested data associatedd features from the the
    input buffered data.
    """
    # OrderedDict where each key corresponds to a type of MRI data (or modality), 
    # and the value contains information about its shape and size.
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
    # This dictionary defines the mask settings for each modality
    # path = The path to the mask file (this can be None if no mask is needed).
    # thr = The threshold for the mask. Values below this threshold will be masked out (set to 0).
    MASKS = {
        "vbm": {
            "path": None,
            "thr": 0.05},
        "quasiraw": {
            "path": None,
            "thr": 0}
    }

    # initializes the FeatureExtractor object
    # dtype = The data type (modality) to work with, e.g., "vbm", "quasiraw", etc.
    # mock: A flag to simulate the transformation without actually loading the data (useful for debugging).
    def __init__(self, dtype, mock=False):
        """ Init class.
        Parameters
        ----------
        dtype: str
            the requested data: 'vbm', 'quasiraw', 'vbm_roi', 'desikan_roi',
            'destrieux_roi' or 'xhemi'.
        """
        if dtype not in self.MODALITIES:
            raise ValueError("Invalid input data type.")
        self.dtype = dtype

        data_types = list(self.MODALITIES.keys())
        index = data_types.index(dtype)
        
        # calculates the cumulative sum of sizes for each modality. 
        # This helps determine where the current modality starts and stops in the input data
        cumsum = np.cumsum([item["size"] for item in self.MODALITIES.values()])
        
        # slice the data corresponding to the selected modality
        if index > 0:
            self.start = cumsum[index - 1]
        else:
            self.start = 0
        self.stop = cumsum[index]
        
        # creates a dictionary of masks with paths for each modality. For "vbm" and "quasiraw", paths to specific mask files are assigned
        # mask files are used to filter the MRI data (i.e., zeroing out areas of the brain that are not of interest)
        self.masks = dict((key, val["path"]) for key, val in self.MASKS.items())
        self.masks["vbm"] = "./data/masks/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
        self.masks["quasiraw"] = "./data/masks/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"

        # If mock is True, the method exits early and skips further processing, 
        # which means no actual loading or transformation of data occurs.
        # (useful for debugging or testing)
        self.mock = mock
        if mock:
            return

        # iterates over the defined mask paths. For each mask:
        for key in self.masks:
            # It checks if the mask file exists.
            if self.masks[key] is None or not os.path.isfile(self.masks[key]):
                raise ValueError("Impossible to find mask:", key, self.masks[key])
            # Loads the mask using nibabel.load() and gets the mask data with .get_fdata().
            arr = nibabel.load(self.masks[key]).get_fdata()
            # Applies the threshold (thr): values below the threshold are set to 0, and values above the threshold are set to 1, essentially creating a binary mask.
            thr = self.MASKS[key]["thr"]
            arr[arr <= thr] = 0
            arr[arr > thr] = 1
            # self.masks[key] = nibabel.Nifti1Image(arr.astype(int), np.eye(4))
            # The mask is then stored as a Nifti1Image object, which is used to apply the mask to the MRI data later.
            self.masks[key] = nibabel.Nifti1Image(arr.astype(np.int32), np.eye(4))


    def fit(self, X, y):
        return self

    # transform() method performs the actual transformation of the input data (X).
    def transform(self, X):
        # If mock is True, it reshapes the data to match the specified shape for the selected modality. 
        # This is used for testing or simulation without applying real transformations.
        if self.mock:
            #print("transforming", X.shape)
            data = X.reshape(self.MODALITIES[self.dtype]["shape"])
            #print("mock data:", data.shape)
            return data
        
        # print(X.shape)
        # slices the input data X according to the modality’s start and stop indices
        select_X = X[self.start:self.stop]
        # specific modalities ("vbm", "quasiraw"), it applies the mask using the unmask() function, 
        # which masks out the irrelevant parts of the MRI data based on the binary mask.
        if self.dtype in ("vbm", "quasiraw"):
            im = unmask(select_X, self.masks[self.dtype])
            # get_fdata() method of a Nifti1Image object retrieves the image data as a numpy array. 
            # This data contains the intensity values of the MRI image, with the mask applied, 
            # which means irrelevant or unimportant regions (those masked out) will have been set to 0 or removed.
            select_X = im.get_fdata()
            # data is transposed to ensure it matches the required shape (depth, height, width) for the MRI data
            select_X = select_X.transpose(2, 0, 1)
        # after unmasking and transposing, the data is reshaped to match the required shape for the selected modality.
        select_X = select_X.reshape(self.MODALITIES[self.dtype]["shape"])
        # print('transformed.shape', select_X.shape)
        return select_X


if __name__ == '__main__':
    import sys
    from torchvision import transforms
    # Imports custom transformations Crop and Pad from a local module called transforms. 
    # These transformations are defined in transforms.py
    from .transforms import Crop, Pad

    # This initializes an instance of the FeatureExtractor class with the modality "vbm". 
    # The FeatureExtractor class is responsible for selecting and transforming the relevant features from the MRI data based on the modality (in this case, "vbm").
    selector = FeatureExtractor("vbm")

    # This method processes the input x (MRI data), likely extracting relevant features and transforming it according to the vbm modality.
    T_pre = transforms.Lambda(lambda x: selector.transform(x))
    # used to chain multiple transformations together into a single transformation pipeline
    T_train = transforms.Compose([
        T_pre,
        Crop((1, 121, 128, 121), type="random"),
        Pad((1, 128, 128, 128)),
        transforms.Lambda(lambda x: torch.from_numpy(x)),
        transforms.Normalize(mean=0.0, std=1.0)
    ])

    train_loader = torch.utils.data.DataLoader(OpenBHB(sys.argv[1], train=True, internal=True, transform=T_train),
                                               batch_size=3, shuffle=True, num_workers=1,
                                               persistent_workers=True)
    
    # reates an iterator over the train_loader DataLoader object, and next() fetches the next batch of data from the iterator.
    x, y1, y2 = next(iter(train_loader))
    print(x.shape, y1, y2)