"""
Contains general purpose utility functions for the experiment
"""

################################################################################
# IMPORTS
###############################################################################
import os
import sys
import torch
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict


################################################################################
# SET SEED
################################################################################
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

################################################################################
# COUNT MODEL PARAMS
################################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


################################################################################
# EARLY STOPPING
################################################################################
class EarlyStopping():
    def __init__(self, patience, delta, verbose, save_model_func, model_path):

        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.save_func = save_model_func
        self.model_path = model_path


    def track(self, val_loss, model):

        if self.best_score is None:
            self.best_score = val_loss
            self.save_func(model, self.model_path, self.verbose)

        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.save_func(model, self.model_path, self.verbose)
            self.counter = 0
        
        else:
            self.counter += 1
            if self.verbose:
                print(f'Early Stopping Count: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


################################################################################
# SAVE CHECKPOINT 
################################################################################
def save_checkpoint(epoch, params, framework, optimiser, lr_sched):
    checkpoint = {
        'epoch':epoch,
        'params': params,
        'framework': framework.state_dict(),
        'optimiser': optimiser.state_dict(),
        'lr_sched': lr_sched.state_dict()
    }

    torch.save(checkpoint, params['base']['cache_path'] + '.pth')

################################################################################
# LOAD CHECKPOINT 
################################################################################
def load_checkpoint(framework, optimiser, lr_sched, checkpoint):
    start_epoch = checkpoint['epoch']
    params = checkpoint['params']
    lr_sched.load_state_dict(checkpoint['lr_sched'])
    framework.load_state_dict(checkpoint['framework'])
    optimiser.load_state_dict(checkpoint['optimiser'])

    return start_epoch, params, framework, optimiser, lr_sched

################################################################################
# SAVE ENCODER
################################################################################
def save_backbone(framework, path, verbose=False):
    """
    Saves the backbone encoder from the framework being used
    """
    backbone = framework.backbone
    torch.save(backbone.state_dict(), path + '.pt')
    if verbose:
        print(f'Successfully Saved Model to: {path}')

################################################################################
# SAVE BOTH THE ENCODER AND FULL FRAMEWORK
################################################################################
def save_both(framework, path, verbose=False):
    """
    Saves the backbone encoder from the framework being used
    """
    backbone = framework.backbone
    torch.save(backbone.state_dict(), path + '_BACKBONE' + '.pt')
    if verbose:
        print(f'Successfully Saved Backbone Model to: {path}')

    torch.save(framework.state_dict(), path + '_FULL' + '.pt')
    if verbose:
        print(f'Successfully Saved Full Model to: {path}')

################################################################################
# LOAD ARBITRARY MODEL/FRAMEWORK
################################################################################
def load_backbone(model, path, verbose=False):
    """
    Loads some arbitrary models' state dict from path
    """
    state_dict = torch.load(path, map_location='cpu')

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[2:] # Removes number at beginning 
    #     new_state_dict[name] = v

    model.load_state_dict(state_dict)
    if verbose:
        print(f'Successfully Loaded Model from: {path}')
    return model

