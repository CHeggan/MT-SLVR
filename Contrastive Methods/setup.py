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

import torch
import numpy as np
import torch.optim as optim

from fit import fit_func
from utils import load_checkpoint

from Frameworks_.cpc import CPC
from Frameworks_.simsiam import SimSiam
from Frameworks_.simclr import SimCLR_V1

from aug_functions import select_possible_augs
from Models_.encoder_selection import resnet_selection
from Dataset_.batching_functions import basic_flexible_prep_batch
from Dataset_.dataset_classes import Sample_Plus_Label_Dataset, FastDataLoader
from Dataset_.dataset_utils import batch_to_log_mel_spec, batch_to_log_mel_spec_plus_stft, nothing_func

################################################################################
# SETUP FUNCTION
################################################################################
def setup_func(params, aug_params, device, restart=False, checkpoint_data=None):

    #########################
    # DATASET & LOADER
    #########################
    print('\nSetting Up Data ... \n')
    data_name = params['data']['name']

        #########################
        # DATASET != METAAUDIO
        #########################
    if data_name != 'METAAUDIO':
        # Grabs name of dataset we are interested in
        
        data_files_path = os.path.join('Dataset_', data_name)

        # Grabs all required dataset specific files, including file names for train and val
        train_files = np.load( os.path.join(data_files_path, params[data_name]['train_files']) + '.npy' )
        val_files = np.load( os.path.join(data_files_path, params[data_name]['val_files']) + '.npy')

        # If we want to use global stats, we have to load them in
        if params['data']['norm'] == 'global':
            stats = np.load(os.path.join(data_files_path, params[data_name]['stats'] + '.npy'))
        else:
            stats = None


        #########################
        # DATASET = METAAUDIO
        #########################
    elif data_name == 'METAAUDIO':
        stats_list = []
        all_train_files = []
        all_val_files = []

        for dataset_name in params[data_name]['datasets']:

            data_files_path = os.path.join('Dataset_', dataset_name)
            # Grabs all required dataset specific files, including file names for train and val
            train_files = np.load( os.path.join(data_files_path, params[dataset_name]['train_files']) + '.npy' )
            val_files = np.load( os.path.join(data_files_path, params[dataset_name]['val_files']) + '.npy')

            # If we want to use global stats, we have to load them in
            if params['data']['norm'] == 'global':
                stats = np.load(os.path.join(data_files_path, params[dataset_name]['stats'] + '.npy'))
            else:
                stats = None
            root_dataset_path = params[dataset_name]['path'] + os.sep

            train_files = np.array(train_files, dtype='U256')
            print(dataset_name, len(train_files))
            train_files = np.char.add(root_dataset_path, train_files)

            val_files = np.array(val_files, dtype='U256')
            val_files = np.char.add(root_dataset_path, val_files)

            all_train_files.append( train_files )
            all_val_files.append( val_files )
            stats_list.append(torch.tensor(stats))

        stats = torch.stack(stats_list)
        stats = stats.mean(dim=0)


        train_files = [item for sublist in all_train_files for item in sublist]
        val_files = [item for sublist in all_val_files for item in sublist]
    

    print(len(train_files))
    print(len(val_files))

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
    model = resnet_selection(dims=params['model']['dims'], model_name=params['model']['name'], 
        fc_out=params['model']['encoder_fc_dim'], in_channels=params['model']['in_channels'])

    #########################
    # FRAMEWORK SELECTION
    #########################
    # Selects the contrastive framework to use
    #   as of now we hardcode the dimensionalities used in original papers
    # We also grab the framework specific optimisation settings and insert into 
    #   'hyper' part of params
    if params['training']['framework'] == 'simsiam':
        framework = SimSiam(model, final_out_dim=2048)
    elif params['training']['framework'] == 'simclrv1':
        framework = SimCLR_V1(model, final_out_dim=256)
    elif params['training']['framework'] == 'CPC': 
        framework = CPC(model, gar_hidden=256, batch_size=params['training']['batch_size'],
            device=device)

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
