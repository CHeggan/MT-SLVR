"""
Script contains main fit function and generalised train/validation loop
"""

###############################################################################
# IMPORTS
###############################################################################
import os 
import sys
import time
import torch
import warnings
import numpy as np
from tqdm import tqdm, trange
from Models_.encoder_selection import resnet_selection
from aug_functions import gen_aug_funcs, apply_aug_funcs
from utils import save_checkpoint, save_backbone, load_backbone, EarlyStopping

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# warnings.filterwarnings("ignore")

###############################################################################
# MAIN FITTING FUNCTION
###############################################################################
def fit_func(framework, prep_batch_fns, dataloaders, optimiser, lr_sched, params, possible_augs, device):

    print('\nStarting Fitting ... \n')
    
    # Unpack variables
    train_loader, valid_loader = dataloaders
    prep_batch, additional_batch_work = prep_batch_fns

    # Creates epoch loop
    epoch_loop = trange(1, params['training']['epochs']+1, file=sys.stdout,
                            desc='Main Loop')
    # Set up value display for the main loop
    epoch_loop.set_postfix({'Train Loss': 0, 'Val Loss': 0})

    # Creates path for teh best model on validation data
    best_model_path = os.path.join(params['base']['results_path'], params['base']['experiment_name'])

    # Initialise the early stopping and model saving class
    early = EarlyStopping(patience=params['early_stopping']['patience'], 
                                    delta=params['early_stopping']['delta'], 
                                    verbose=params['early_stopping']['verbose'], 
                                    save_model_func=save_backbone, 
                                    model_path=best_model_path)

    # Dont necessarily want to go over the full trainset at once
    # We create our first shuffled iteration of the dataloader object
    train_iter = iter(train_loader)

    # MAIN LOOP
    for epoch in epoch_loop:

        framework.train()
        train_loss = 0
        # # Perform val_gap number of training steps 
        # for idx in range(params['training']['val_gap']):
        #     # As we are moving through the dataloader object batch by batch using
        #     #   iter and next. When current iter runs out, will throw an exception
        #     # We simply create a new re-shuffled loader when this happens 
        #     try:
        #         batch = next(train_iter)
        #     except StopIteration:
        #         train_iter = iter(train_loader)
        #         batch = next(train_iter)

        for idx, batch in enumerate(train_loader):

            # Clear gradients so that dont avg over batches
            optimiser.zero_grad()

            prep_time_start = time.time()
            x, y= prep_batch(batch)
            prep_time_finish = time.time()
            # print(prep_time_finish-prep_time_start)

            augs_to_apply = gen_aug_funcs(possible_augs=possible_augs,
                params=params,
                batch_size=params['training']['batch_size'],
                batch_wise=params['aug_control']['batch_wise'])

            x = apply_aug_funcs(samples=x, 
                augs_to_apply=augs_to_apply,
                batch_wise=params['aug_control']['batch_wise'])

            # If we use simclr or simsiam, we need to contrast two views of data
            if params['training']['framework'] in ['simsiam', 'simclrv1']:
                # Converts to specs etc if needed
                x[0] = additional_batch_work(x[0], params['ft_params'])
                x[1] = additional_batch_work(x[1], params['ft_params'])

                loss = framework.forward(x[0], x[1])

            # If we use cpc, we dont need multi-views
            elif params['training']['framework'] in ['CPC']:
                x = additional_batch_work(x[0], params['ft_params'])
                loss, accuracy, z, c = framework.forward(x)

            loss.backward()
            optimiser.step()
            lr_sched.step()

            train_loss += loss.item()

        # Do a validation pass over full val split
        framework.eval()
        val_loss = 0
        for idx, batch in enumerate(valid_loader):
            x, y = prep_batch(batch)

            x = apply_aug_funcs(samples=x, 
                augs_to_apply=augs_to_apply,
                batch_wise=params['aug_control']['batch_wise'])

            # If we use simclr or simsiam, we need to contrast two views of data
            if params['training']['framework'] in ['simsiam', 'simclrv1']:
                # Converts to specs etc if needed
                x[0] = additional_batch_work(x[0], params['ft_params'])
                x[1] = additional_batch_work(x[1], params['ft_params'])

                loss = framework.forward(x[0], x[1])

            # If we use cpc, we dont need multi-views
            elif params['training']['framework'] in ['CPC']:
                x = additional_batch_work(x[0], params['ft_params'])
                loss, accuracy, z, c = framework.forward(x)
            
            val_loss += loss.item()

        # Calculate avg values for train and validation loss
        avg_train_loss = train_loss/params['training']['val_gap']
        avg_val_loss = val_loss / len(valid_loader)

        # Best model tracking and saving
        # Determines whether we have a new best model based on val loss
        early.track(avg_val_loss, framework)


        # Write avg losses to tensorboard
        writer.add_scalar("Loss/avg_train", avg_train_loss, epoch)
        writer.add_scalar("Loss/avg_val", avg_val_loss, epoch)

        # Update visual loop bar
        loss_dict = {'Train Loss': round(avg_train_loss, 2),
             'Val Loss': round(avg_val_loss, 2)}
        epoch_loop.set_postfix(loss_dict)


        # Saves checkpoint for restart of needed, this is not best model saving
        #save_checkpoint(epoch, params, framework, optimiser, lr_sched)
        
        if early.early_stop:
            print('EARLY STOPPING TRIGGERED')
            break 


    writer.flush()
    writer.close()

    timestr = time.strftime("%d%m%Y-%H%M%S")
    print(f'Finished Training Model at {timestr}')

    print(f'Trial Load Starting')
    best_model = resnet_selection(dims=params['model']['dims'], model_name=params['model']['name'], 
        fc_out=params['model']['encoder_fc_dim'], in_channels=params['model']['in_channels'])
    best_model = load_backbone(best_model, best_model_path, verbose=True)


