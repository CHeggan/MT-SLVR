"""
High level model selection script. Main function is pulled from setup.py
"""

################################################################################
# IMPORTS
################################################################################
import numpy as np

from Models_.encoder_selection import resnet_selection, adapter_resnet_selection, split_resnet_selection


################################################################################
# IMPORTS
################################################################################
def model_selection(adapter_type, model_name, dims, in_channels, fc_out, fc_splits):
    """High level neural model/backbone selector.

    Args:
        adapter_type (str): The type of adapter to use in the model
        model_name (str): The name of the base neural model being used, i.e resnet18
        dims (int): The dimensionality of input samples (1d raw audio) or 2d
            (spectrograms)
        in_channels (int): Number of input channels in the input data
        fc_out (int): The number of neurons in the final fully connected layer
        fc_splits (int): Number of splits in the final fully connected layer
    """
    # We select our model based on what adapters we are using
    if adapter_type == 'None':
        model = resnet_selection(dims=dims, model_name=model_name, 
            fc_out=fc_out, in_channels=in_channels)
        fc_list = [fc_out]

    elif adapter_type in ['bn', 'series', 'parallel']:
        model_name = 'adapter_' + model_name

        fc_list = [int(np.floor(fc_out/fc_splits))]*fc_splits

        model = adapter_resnet_selection(dims=dims,
            fc_out_list=fc_list,
            in_channels=in_channels,
            task_mode=adapter_type,
            num_tasks=fc_splits,
            model_name=model_name)

    elif adapter_type == 'split':
        model_name = 'split_' + model_name
        fc_list = [int(np.floor(fc_out/fc_splits))]*fc_splits

        model = split_resnet_selection(dims=dims,
            fc_out=fc_list,
            in_channels=in_channels,
            model_name=model_name)
        
    return model
