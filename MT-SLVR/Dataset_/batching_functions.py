"""
Script contains prep batch functions
"""
###############################################################################
# IMPORTS
###############################################################################
import sys
import torch
import numpy as np

###############################################################################
# BASIC & FLEXIBLE VARIABLE LENGTH FRIENDLY PREP BATCH FUNCTION
##############################################################################
def basic_flexible_prep_batch(device, data_type='float', input_size=None, variable=False):
    """Is the parent function for batch prepping. Returns an initialised function 

    Args:
        device (torch CUDA object): The CUDA device we want to load data to
        data_type (str, optional): The type to convert the input data to, can be 
            float or double
        input_size (int, optional): The expected size of input to the model. This allows us 
            to split up data and stack it if is needed 
        variable (boolean, optional): Whether the data incoming is variable length 
            samples or not
    """
    def basic_flexible_prep_batch(batch):
        """The child prep batch function. Takes some batch and processes it
            into proper tasks before moving it to a GPU for calculations.
        Depending on whether the data is variable length or not, batch data will
            come in as lists not tensors

        Args:
            batch (Tuple of lists or Tensors): The unformatted batch of data and tasks

        Returns:
            Tensor, Tensor: The formatted x, y tensor pairs
        """
        x, y = batch

        # If variable length, we split up and stack on top of one another
        if variable:
            all_tensors = []
            lengths = []
            for sample_set in x:
                if sample_set.ndim < 3:
                    sample_set = sample_set.unsqueeze(1)
                # Track the number of sub-samples per clip
                lengths.append(sample_set.shape[0])
                all_tensors.append(sample_set)

            x = torch.cat(all_tensors)
            y = torch.stack(y)
        
            if data_type == 'float':
                x = x.float().to(device)
                y = y.float().to(device)
            elif data_type == 'double':
                x = x.double().to(device)
                y = y.double().to(device)
            else:
                raise ValueError('data type not recognised')

            return x, y, lengths

        else:
            if data_type == 'float':
                x = x.float().to(device)
                y = y.float().to(device)
            elif data_type == 'double':
                x = x.double().to(device)
                y = y.double().to(device)
            else:
                raise ValueError('data type not recognised')

            return x, y
    return basic_flexible_prep_batch
