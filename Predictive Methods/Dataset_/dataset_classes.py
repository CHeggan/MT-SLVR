"""
Script contains general classes for dataset sampling. 

Due to how large most of the datasets are, we employ lazy sampling, that is 
    grabbing a sample only when requested. This is generally more CPU heavy but
    is our only real option for large high-dimensional datasets. 
"""

###############################################################################
# IMPORTS
###############################################################################
from cProfile import label
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from torch.utils.data import Dataset
from Dataset_.dataset_utils import per_sample_scale, nothing_func, given_stats_scale

###############################################################################
# SAMPLE PLUS LABEL DATASET (SAMPLED FILES CONTAIN BOTH THE SIGNAL AND ONE-HOT)
###############################################################################
class Sample_Plus_Label_Dataset(Dataset):
    def __init__(self, data_path, norm, variable=False, eval=False, expected_files=None, 
        class_weights=None, stats=None, df_path=None):
        """A basic sample plus label dataset (i.e both the sample and label vector 
            is stored in the path). The class doesn't natively rely on having access 
            to class data, this makes the default implementation very fast and efficient. 
            We do however add in the option for dataframe class data to be included, 
            allowing for more complex sampling and class-dependant augmentation techniques. 

            This class also natively deals with variable length sets, at both training
            and evaluation times. 

        Args:
            data_path (str): The exact path to the dataset folder of interest
            norm (str): Type of normalisation to apply, options found in get_norm_func
            variable (bool, optional): Whether the dataset is variable length. Defaults to False.
            eval (bool, optional): Whether the data we are using is for eval, only 
                important if variable length. Defaults to False.
            expected_files (list/array, optional): If not using the whole path of 
                files available, list of files to use. Defaults to None.
            class_weights (arr, optional): Class weights 
                weights . Defaults to None.
            stats (2d arr, optional): global normalisation statistics to use [mu, sigma]
                to use if 'global'. Defaults to None.
            df_path (str, optional): Path to label and id dataframe
        """        
        self.norm = norm
        self.data_path = data_path
        # Dont have class weight file for val or testing sets
        self.class_weights = class_weights

        if df_path != None:
            # Loads class id and label dataframe and splits them into the useful parts
            #   This can now be used for things like class-aware mix-up or other techniques
            df = pd.read_csv(df_path, index_col=0)
            self.ids = df['id'].to_numpy(dtype=str)

            # Accounts for fsd50k meta data file, having a split column at end of df
            try:
                df = df.drop('split', axis=1)
            except:
                pass

            labels = df.iloc[:, 1:]
            self.labels = labels.to_numpy(dtype=int)

        # Assumes all files in the dataset are present directly in the data path
        #   fine assumption for actual experiments but annoying for testing
        self.expected_files = expected_files
        # If dataset is variable length (i.e FSD50k), we perform random selection
        #   of available sub-sample clips, If variable and eval set we want to 
        #   keep all samples and take some vote at inference
        self.variable = variable
        self.eval = eval

        # If we arent given expected files, we just use all of teh files available
        if np.all(expected_files == None):
            self.expected_files = os.listdir(self.data_path)

        # Set the normalisation function needed, default to not need stat path
        self.norm_func = self.set_norm_func(norm, stats=stats)




    def set_norm_func(self, norm, stats):
        """Grabs the relevant normalisation function to use when getting data

        Args:
            norm (str): The normalisation types to use
            stats_file (str): Path to the global norm stats of data being used

        Raises:
            ValueError: If normalisation type not recognised, inform user

        Returns:
            function: The normalisation function to call over incoming data
        """
        if norm == 'l2':
            norm_func = preprocessing.normalize

        elif norm == 'None':
            norm_func = nothing_func

        elif norm == 'sample':
            norm_func = per_sample_scale

        elif norm == 'global':
            self.mu = stats[0]
            self.sigma = stats[1]
            norm_func = given_stats_scale
        else:
            raise ValueError('Passed norm type unsupported')

        return norm_func

    def __getitem__(self, item):
        # Data contains both sample and label vector
        sample_path = os.path.normpath(os.path.join(self.data_path, str(self.expected_files[item]) + '.pt'))

        # Loads in sample and one-hot encoded labels
        sample, one_hot_labels = torch.load(sample_path)

        # If we are operating with variable train data 
        if self.variable and not self.eval:
            # Sometimes samples that are var length have exactly the expected 
            #   length so come in as [1, SR*expected], dont want to sub-sample these
            if sample.ndim > 2:
                idx = np.random.choice(sample.shape[0])
                sample = sample[idx]

        # Deals with normalisation of various types
        if self.norm in ['global']:
            sample = self.norm_func(sample, self.mu, self.sigma)

        else:
            sample = self.norm_func(sample)

        return sample, one_hot_labels
    

    def __len__(self):
        return len(self.expected_files)


###############################################################################
# STABLE NUM WORKERS DATALOADER
###############################################################################
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
