"""
Includes the main experiment call as well as:
    - experiment param loading
    - model param loading
"""
################################################################################
# IMPORTS
################################################################################
import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np

from utils import set_seed
from setup import setup_func

################################################################################
# MAIN CALL & ARGPARSE  
################################################################################
if __name__ == '__main__':
    #########################
    # ARGPARSE
    #########################
    parser = argparse.ArgumentParser(description='SimCLR/SimSiam/CPC Contrastive Learning Framework')

    parser.add_argument('--framework', required=True, type=str,
        help='Name of contrastive framework to use, simsiam/simclrv1/CPC', choices=['simsiam', 'simclrv1', 'CPC'])

    parser.add_argument('--data_name', required=True, type=str,
        help='Name of the dataset to use for SS training')

    parser.add_argument('--dims', required=True, type=int,
        help='Number of dimensions input data has', choices=[1, 2])

    parser.add_argument('--in_channels', required=True, type=int,
        help='Number of input channels data has to model', choices=[1, 3])

    parser.add_argument('--batch_size', required=True, type=int,
        help='Batch size of training routine, 1d(2080ti=48/3090=128), 2d(2080ti=512/3090=850)')

    parser.add_argument('--lr', required=True, type=float,
        help='Learning rate of the training process')

    parser.add_argument('--p', required=True, type=float,
        help='Probability of Augmentation being applied (individually)')

    parser.add_argument('--model_fc_out', required=True, type=int,
        help='Size of the final fully-connected head in backbone model')

    parser.add_argument('--augs_to_exclude', nargs='*', required=True,
        help='List of augmentations to exclude', 
        choices=['pitch_shift', 'fade', 'white_noise', 'mixed_noise', 'time_masking', 'time_shift', 'time_stretch'])

    args = vars(parser.parse_args())

    #########################
    # PARAM LOADING
    #########################
    # Loads in other expeirment params
    with open("experiment_cfg.yaml") as stream:
        params = yaml.safe_load(stream)

    with open("Dataset_/aug_params.yaml") as stream:
        aug_params = yaml.safe_load(stream)

    
    #########################
    # COMBINE ARGUMENTS
    #########################
    params['training']['framework'] = args['framework']
    params['data']['name'] = args['data_name']

    params['aug_control']['augs_to_exclude'] = args['augs_to_exclude']
    params['training']['batch_size'] = args['batch_size']
    params['hyper']['initial_lr'] = args['lr']

    params['model']['dims'] = args['dims']
    params['model']['in_channels'] = args['in_channels']

    aug_params['p'] = args['p']
    params['model']['encoder_fc_dim'] = args['model_fc_out']
        
    #########################
    # DEVICE SETTING
    #########################
    # Setting of cuda device
    if params['base']['cuda'] == True:
        device = torch.device('cuda')

    elif params['base']['cuda'] == 'cpu':
        device = torch.device('cpu')

    else:
        cuda_int = params['base']['cuda']
        device = torch.device('cuda:' + str(cuda_int))


    #########################
    # PATH REDIRECTION  
    #########################
    # If we are using a single static set (i.e FSD50K), we check a few locations for it
    if args['data_name'] != 'METAAUDIO':
        exists = os.path.isdir(params[args['data_name']]['path'])
        if exists:
            pass
        elif exists == False:
            params[args['data_name']]['path'] = params[args['data_name']]['backup_path']

    #########################
    # SEED SETTING
    #########################
    if params['base']['seed'] == 'random':
        seed = np.random.randint(low=0, high=1000)
        params['base']['seed'] = seed
    else:
        seed = params['base']['seed']
        
    set_seed(seed)

    #########################
    # SET CACHE & RESULTS PATH
    #########################
    # Here we set a cache path, something we rely on if need to restart
    # experiment_name = params['base']['experiment_name'] + '_' \
    #     + params['data']['norm'] + '_' + params['data']['name'] + '_' +\
    #         params['model']['name']

    # Get time date for 
    timestr = time.strftime("%d%m%Y-%H%M%S")
    # experiment_name = params['training']['framework'] + '_' + params['base']['experiment_name'] \
    #     + '_' + params['data']['name'] + '_' + timestr

    # Temp expeirment name
    aug_name_part = ''
    for name in params['aug_control']['augs_to_exclude']:
        aug_name_part += name + '__'

    experiment_name = params['data']['name'] + '_' + params['training']['framework'] + '_' + str(params['model']['dims']) + \
        '_' + str(params['model']['in_channels']) + '_' +  aug_name_part + '_' + str(params['training']['batch_size'])+ \
            '_' + str(params['hyper']['initial_lr']).replace('.', '_') + '_' + timestr


    cache_path = os.path.join('CACHE_', experiment_name)
    results_path = os.path.join('RESULTS_', experiment_name)

    # Making (or trying to) teh appropriate results folder
    try:
        os.mkdir(results_path)
        print(f'Created results directory: {results_path}')
    except:
        print(f'Some Issue creating directory: {results_path}')

    # Store the new paths etc
    params['base']['cache_path'] = cache_path
    params['base']['results_path'] = results_path
    params['base']['experiment_name'] = experiment_name

    # Saves a copy of the expeirment params to the results folder
    with open(os.path.join(results_path, 'params.yaml'), 'w') as file:
        documents = yaml.dump(params, file)

    #########################
    # RUN EXPERIMENT
    #########################

    final_results = setup_func(params=params, 
                                aug_params=aug_params, 
                                device=device)



    
            



