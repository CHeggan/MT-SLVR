"""
File contains the necessary classes for the CPC learning module, builds heavily
    on: https://github.com/Spijkervet/contrastive-predictive-coding/
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from .infonce import InfoNCE

import sys
###############################################################################
# AUTOREGRESSOR MODULE
###############################################################################
class Autoregressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoregressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)
        return out

###############################################################################
# CPC FRAMEWORK
###############################################################################
class CPC(nn.Module):
    def __init__(self, backbone, gar_hidden, batch_size, device):
        super(CPC, self).__init__()

        self.device = device

        """
        First, a non-linear encoder genc maps the input sequence of observations xt to a
        sequence of latent representations zt = genc(xt), potentially with a lower temporal resolution.
        """
        self.backbone = backbone

        """
        We then use a GRU RNN [17] for the autoregressive part of the model, gar with 256 dimensional hidden state.
        """
        self.autoregressor = Autoregressor(input_dim=1, hidden_dim=gar_hidden)


        cpc_args = {'learning_rate': 2.0e-4,
                    'prediction_step': 12,
                    'negative_samples':10,
                    'subsample':True,
                    'calc_accuracy': False,
                    'batch_size': batch_size}

        self.loss = InfoNCE(gar_hidden=gar_hidden, genc_hidden=1, cpc_args=cpc_args)

    def get_latent_size(self, input_size):
        x = torch.zeros(input_size).to(self.device)

        z, c = self.get_latent_representations(x)
        return c.size(2), c.size(1)


    def get_latent_representations(self, x):
        """
        Calculate latent representation of the input with the encoder and autoregressor
        :param x: inputs (B x C x L)
        :return: loss - calculated loss
                accuracy - calculated accuracy
                z - latent representation from the encoder (B x L x C)
                c - latent representation of the autoregressor  (B x C x L)
        """

        # calculate latent represention from the encoder
        z = self.backbone(x).unsqueeze(1)

        z = z.permute(0, 2, 1)  # swap L and C
        # print(z.shape)
        # calculate latent representation from the autoregressor
        c = self.autoregressor(z)
        
        # print('z check', z.shape)
        # print('c check', c.shape)
        # TODO checked
        return z, c


    def forward(self, x):
        z, c = self.get_latent_representations(x)
        # print('z in latent', z.shape)
        loss, accuracy = self.loss.get(x, z, c)
        # print(loss)
        #sys.exit()
        return loss, accuracy, z, c