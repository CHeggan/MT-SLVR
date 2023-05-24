"""
Script contains all of the experiment setup, includes things like:
    - datasets, dataloaders
    - batching function initialisation
    - model loading
    - loss functions and optmisers
    - call to fitting function
"""

################################################################################
# IMPORTS
################################################################################
import os
import sys

import numpy as np
import torch.optim as optim

from fit import fit_func
from utils import load_checkpoint

from Frameworks_.mtl import MTL

from aug_functions import select_possible_augs
from Dataset_.batching_functions import basic_flexible_prep_batch
from Dataset_.dataset_classes import Sample_Plus_Label_Dataset, FastDataLoader
from Dataset_.dataset_utils import batch_to_log_mel_spec, batch_to_log_mel_spec_plus_stft, nothing_func
from Models_.encoder_selection import resnet_selection, adapter_resnet_selection, split_resnet_selection

################################################################################
# SETUP FUNCTION
################################################################################
def setup_func(params, aug_params, device, restart=False, checkpoint_data=None):

    #########################
    # DATASET & LOADER
    #########################
    print('\nSetting Up Data ... \n')
    # Grabs name of dataset we are interested in
    data_name = params['data']['name']
    data_files_path = os.path.join('Dataset_', data_name)

    # Grabs all required dataset specific files, including file names for train and val
    train_files = np.load( os.path.join(data_files_path, params[data_name]['train_files']) + '.npy' )
    val_files = np.load( os.path.join(data_files_path, params[data_name]['val_files']) + '.npy')

    # If we want to use global stats, we have to load them in
    if params['data']['norm'] == 'global':
        stats = np.load(os.path.join(data_files_path, params[data_name]['stats'] + '.npy'))
    else:
        stats = None


    # Defines the dataset to be used
    dataset = Sample_Plus_Label_Dataset

    trainset = dataset(data_path=params[data_name]['path'],
                        norm=params['data']['norm'],
                        variable=params[data_name]['variable'],
                        eval=False,
                        expected_files=train_files,
                        stats=stats,
                        ext=params[data_name]['ext'],
                        label=params[data_name]['label'])

    validset = dataset(data_path=params[data_name]['path'],
                        norm=params['data']['norm'],
                        variable=params[data_name]['variable'],
                        eval=False,
                        expected_files=val_files,
                        stats=stats,
                        ext=params[data_name]['ext'],
                        label=params[data_name]['label'])

    # Sets up all of our dataset loaders, we just use basic loaders for now
    train_loader = FastDataLoader(trainset, batch_size=params['training']['batch_size'],
        num_workers=params['training']['num_workers'], shuffle=True)

    valid_loader = FastDataLoader(validset, batch_size=params['training']['batch_size'],
        num_workers=params['training']['num_workers'],  shuffle=True)

    #########################
    # BATCHING FUNCTIONS
    #########################
    prep_batch = basic_flexible_prep_batch(device=device, data_type='float')

    if params['model']['dims'] == 1 and params['model']['in_channels'] ==1:
        extra_batch_work = nothing_func
    elif params['model']['dims'] == 2 and params['model']['in_channels'] ==1:
        extra_batch_work = batch_to_log_mel_spec
    elif params['model']['dims'] == 2 and params['model']['in_channels'] ==3:
        extra_batch_work = batch_to_log_mel_spec_plus_stft
    else:
        raise ValueError('Thaaaaaaanks, an incorrect config file')

    #########################
    # AUGMENTATIONS
    #########################
    # WE DONT PUT TRANSFORMS STRAIGHT INTO THE DATASET CLASS AS WE DONT 
    #   ALWAYS WANT TO APPLY THEM. i.e DONT WANT FOR PREDICTIVE TASKS IN SAME
    #   WAY AS CONTRASTIVE 
    
    # Do something with this later
    aug_params = aug_params

    possible_augs = select_possible_augs(params['aug_control']['augs_to_exclude'], 
        aug_params=aug_params, sample_rate=16000)


    #########################
    # MODEL SELECTION
    #########################
    # We select our model based on what adapters we are using
    if params['adapters']['task_mode'] == 'None':
        model = resnet_selection(dims=params['model']['dims'], model_name=params['model']['name'], 
            fc_out=params['model']['encoder_fc_dim'], in_channels=params['model']['in_channels'])
        fc_list = [params['model']['encoder_fc_dim']]

    elif params['adapters']['task_mode'] in ['bn', 'series', 'parallel']:
        params['model']['name'] = 'adapter_' + params['model']['name']

        fc_list = [int(np.floor(params['model']['encoder_fc_dim']/params['adapters']['num_splits']))]*params['adapters']['num_splits']

        model = adapter_resnet_selection(dims=params['model']['dims'],
            fc_out_list=fc_list,
            in_channels=params['model']['in_channels'],
            task_mode=params['adapters']['task_mode'],
            num_tasks=params['adapters']['num_tasks'],
            model_name=params['model']['name'])

    elif params['adapters']['task_mode'] == 'split':
        params['model']['name'] = 'split_' + params['model']['name']
        fc_list = [int(np.floor(params['model']['encoder_fc_dim']/params['adapters']['num_tasks']))]*params['adapters']['num_tasks']

        model = split_resnet_selection(dims=params['model']['dims'],
            fc_out=fc_list,
            in_channels=params['model']['in_channels'],
            model_name=params['model']['name'])


    #########################
    # FRAMEWORK SELECTION
    #########################
    if params['adapters']['num_splits'] == 1:
        backbone_cont_size = fc_list[0]
        backbone_pred_size = fc_list[0]
    
    elif params['adapters']['num_splits'] == 2:
        backbone_cont_size = fc_list[0]
        backbone_pred_size = fc_list[1]

    else:
        raise ValueError(f"Num fc layer splits given as {params['adapters']['num_splits']}\
        , please give value of either 1 or 2")

    framework = MTL(backbone=model,
        cont_method=params['training']['cont_framework'],
        pred_method=params['training']['pred_framework'],
        pred_output=len(possible_augs),
        backbone_cont_size=backbone_cont_size,
        backbone_pred_size=backbone_pred_size,
        pred_weight=params['training']['pred_weight'],
        adapter=params['adapters']['task_mode'],
        batch_size=params['training']['batch_size'])

    framework = framework.to(device)

    #########################
    # OPTIMISER & SCHEDULER
    #########################
    optimiser = optim.Adam(framework.parameters(), lr=float(params['hyper']['initial_lr']),
        weight_decay=params['hyper']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=params['training']['epochs'], 
        eta_min=0, last_epoch=-1, verbose=False)

    #########################
    # COMBINE ALL PARAMS
    #########################
    # We combine all parameters here for easier reference later
    params['aug_params'] = aug_params

    #########################
    # LOAD CHECKPOINT
    #########################
    if restart:
        print('\nRestarting ... \n')
        start_epoch, params, framework, optimiser, scheduler = \
            load_checkpoint(framework, optimiser, scheduler, checkpoint_data)

        params['training']['epochs'] = params['training']['epochs'] - start_epoch

    #########################
    # CALL TO FIT FUNCTION
    #########################

    fit_func(framework=framework,
        prep_batch_fns=[prep_batch, extra_batch_work],
        dataloaders=[train_loader, valid_loader],
        optimiser=optimiser,
        lr_sched=scheduler,
        params=params, 
        possible_augs=possible_augs,
        device=device)
