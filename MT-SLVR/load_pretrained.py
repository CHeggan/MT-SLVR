"""
Script to load pre-trained MT-SLVR models. 

To use this script, pre-trained models will have to be downloaded from here:
    "https://drive.google.com/drive/folders/1j8876DWc04GSxaV0lMvLi69BLlP45w2g?usp=sharing".  

"""
################################################################################
# IMPORTS
################################################################################
import os 

import torch

from utils import load_backbone
from Models_.meta_encoder_selection import model_selection


################################################################################
# MT-SLVR PRE-TRAINED WEIGHTS LOADING
################################################################################
# Important parameters that need to be set for correct loading. These parameters
#   can be inferenced from the paper ot the pre-trained model path. See below a
#   note on how to obtain these params from weights path

model_path = "X:/Trained_model_storage/MT-SLVR IS23/MT_SLVR_MODELS/AS_BAL_simsiam_trans_1_0_parallel_2_3__100_5e-05_21022023-145537.pt"


ADAPTER = "parallel"
NAME = "resnet18"
DIMS = 2
CHANNELS = 3
FC_OUT = 1000
FC_SPLITS = 2


model = model_selection(adapter_type=ADAPTER,
                        model_name=NAME,
                        dims=DIMS,
                        in_channels=CHANNELS,
                        fc_out=FC_OUT,
                        fc_splits=FC_SPLITS)

model = load_backbone(model, model_path)


device = 'cpu'
# Example input data. Spectrogram with 3 channels
data = torch.rand(size=(10, 3, 128, 157))

if FC_SPLITS == 1:
    out = model.forward(data)

    # Outputs tensor of shape (10, 1000)
    print(out.shape)

elif FC_SPLITS == 2:
    out_1 = model.forward(data, task_int=0)
    out_2 = model.forward(data, task_int=1)

    # Outputs tensor of shape (10, 500)
    print(out_1.shape)
    print(out_2.shape)


################################################################################
# READING PRE-TRAINED WEIGHTS PATH
################################################################################
# Output weights follow a basic naming convention which can be used to reconstruct
#   the base model form this code

example_model_path = "AS_BAL_simsiam_trans_1_0_parallel_2_3__100_5e-05_21022023-145537.pt"

# In order as they appear in the string, we can identify the following params

dataset = "AS_BAL" 
contrastive_algorithm = 'simsiam'
predictive_algorithm = 'trans'
predictive_algorithm_weighting = "1_0"
adapter = "parallel" #**
input_data_representation_dimensions = 2 #**
input_channels = 3 #**
batch_size_used_in_train = 100
learning_rate = "5e-05"
date_and_time = "21022023-145537"


# Not all of these are required for the model building. We have added a ** next to those which are needed
# For our paper, all backbone models used as resnet18s so the name isnt explicitly spelled out in the pre-trained weights. 
#   It does however still need to be include when building the model