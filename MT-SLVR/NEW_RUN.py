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
    parser = argparse.ArgumentParser(description='MTL framework between contrastive (SimCLR/SimSiam/CPC) and transformation predictive algorithms')

    # Primary controls over the MTL part of codebase
    parser.add_argument('--cont_framework', required=True, type=str,
        help='Name of contrastive framework to use, simsiam/simclrv1/CPC', choices=['simsiam', 'simclrv1', 'cpc'])

    parser.add_argument('--pred_framework', required=True, type=str,
        help='Name of transformation prediction framework to use, trans/param', choices=['trans', 'param'])

    parser.add_argument('--pred_weight', required=True, type=float,
        help='Weight of the predictive loss function in training')

    parser.add_argument('--adapter', required=True, type=str,
        help='Type of adapter to use for MTL approach', choices=['None', 'split', 'bn', 'series', 'parallel'])
    
    parser.add_argument('--num_splits', required=True, type=int,
            help='Number f splits in final backbone layer. Cannot be <2 with adapter type split')



    # Controls over basic training and model setup
    parser.add_argument('--batch_size', required=True, type=int,
        help='Batch size of training routine, 1d(2080ti=48/3090=128), 2d(2080ti=512/3090=850)')

    parser.add_argument('--lr', required=True, type=float,
        help='Learning rate of the training process')

    parser.add_argument('--model_fc_out', required=True, type=int,
        help='Size of the final fully-connected head in backbone model, assumes we merge task heads, per head we do total/num heads')


    ## Non required inputs (have defaults)
    parser.add_argument('--p', required=False, type=float, default=1.0,
        help='Probability of Augmentation being applied (individually)')

    parser.add_argument('--augs_to_exclude', nargs='*', required=False, default='',
        help='List of augmentations to exclude', 
        choices=['pitch_shift', 'fade', 'white_noise', 'mixed_noise', 'time_masking', 'time_shift', 'time_stretch'])

    parser.add_argument('--data_name', required=False, type=str, default='AS_BAL',
        help='Name of the dataset to use for SS training')

    parser.add_argument('--dims', required=False, type=int, default=2,
        help='Number of dimensions input data has', choices=[1, 2])

    parser.add_argument('--in_channels', required=False, type=int, default=3,
        help='Number of input channels data has to model', choices=[1, 3])

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
    params['training']['cont_framework'] = args['cont_framework']
    params['training']['pred_framework'] = args['pred_framework']
    params['training']['pred_weight'] = args['pred_weight']
    params['adapters']['task_mode'] = args['adapter']
    params['adapters']['num_splits'] = args['num_splits']

    params['training']['batch_size'] = args['batch_size']
    params['hyper']['initial_lr'] = args['lr']
    params['model']['encoder_fc_dim'] = args['model_fc_out']

    
    aug_params['p'] = args['p']
    params['aug_control']['augs_to_exclude'] = args['augs_to_exclude']
    params['data']['name'] = args['data_name']
    params['model']['dims'] = args['dims']
    params['model']['in_channels'] = args['in_channels']

    
    
        
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
        print(cuda_int)
        device = torch.device('cuda:' + str(cuda_int))

    print(device)

    #########################
    # PATH REDIRECTION  
    #########################
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

    # Exper name: Data, cont frame, pre frame, dims, channels, exluded augs, batch size, lr, adapter, pred weight
    experiment_name = params['data']['name'] + '_' + params['training']['cont_framework'] + '_' + params['training']['pred_framework'] \
        + '_' + str(params['training']['pred_weight']).replace('.', '_') + '_' + params['adapters']['task_mode'] + '_' + str(params['model']['dims']) + \
        '_' + str(params['model']['in_channels']) + '_' +  aug_name_part + '_' + str(params['training']['batch_size'])+ \
            '_' + str(params['hyper']['initial_lr']).replace('.', '_') + '_' + timestr

    print(experiment_name)


    results_path = os.path.join('RESULTS_', experiment_name)

    # Making (or trying to) teh appropriate results folder
    try:
        os.mkdir(results_path)
        print(f'Created results directory: {results_path}')
    except:
        print(f'Some Issue creating directory: {results_path}')

    # Store the new paths etc
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



    
            



