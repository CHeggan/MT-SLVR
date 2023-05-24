"""
File contains the multi-task learning class, which shared a base encoder between
    two different loss heads. 

We also merge teh computational graphs by using both x1 and x2 contrastive views 
    in the predictive part of the network and passing all through the main resnet backbone. 

In both simclr and simsiam both batches x1 and x2 appear to contribute batch wise 
    to the gradient calculations. From tihis it makes sense ot utilise both x1 and 
        x2 batches in the predictive part as well. We can do this by simply joining 
        them and repeating the augmentation label y
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from Frameworks_.cpc import CPC
from Frameworks_.infonce import InfoNCE
from Frameworks_.simclr import NT_XentLoss, projection_MLP_SimCLR
from Frameworks_.simsiam import prediction_MLP_SimSiam, projection_MLP_SimSiam, D
from Frameworks_.aug_prediction import multi_label_bce, prediction_MLP

###############################################################################
# SIMPLE LOSS WEIGHTING FUNCS
###############################################################################
def weighted_loss(cont_loss, pred_loss, pred_weight):
    final_loss = cont_loss + pred_weight*pred_loss
    return final_loss


###############################################################################
# IMPORTS
###############################################################################
class MTL(nn.Module):
    def __init__(self, backbone, cont_method, pred_method, backbone_pred_size, backbone_cont_size, 
            pred_output, pred_weight, adapter, batch_size):

        super(MTL, self).__init__()

        self.cont_method = cont_method
        self.pred_method = pred_method

        self.adapter = adapter

        self.pred_weight = pred_weight
        self.loss_combination = weighted_loss


        # Deal with setup of relevant contrastive functions
        if self.cont_method == 'simsiam':
            self.cont_loss = D
            self.cont_projector = projection_MLP_SimSiam(backbone_cont_size,
                num_layers=3, hidden_dim=2048, out_dim=2048)
            self.cont_predictor = prediction_MLP_SimSiam(in_dim=2048, hidden_dim=512, 
            out_dim=2048)

        elif self.cont_method == 'simclrv1':
            self.temp = 0.07
            self.cont_loss = NT_XentLoss
            self.cont_projector = projection_MLP_SimCLR(in_dim=backbone_cont_size, out_dim=256)

        elif self.cont_method == 'cpc':
            self.cpc = CPC(backbone=backbone,
                gar_hidden=256,
                batch_size=batch_size)

        else:
            raise ValueError(f'Contrastive method not recognised: {self.cont_method}')


        # Deal with setup of relevant predictive functions
        if self.pred_method == 'trans':
            self.pred_loss = multi_label_bce
            # Projection module for aug prediction is 4 layers
            self.predict_projector = prediction_MLP(backbone_pred_size, num_layers=4,
                hidden_dims=[256, 64, 16], out_dim=pred_output)

        elif self.pred_method == 'param':
            raise ValueError('Not set up yet')

        else:
            raise ValueError(f'Predictive method not recognised')


        # When inferencing, only the backbone with its fc layer is used
        self.backbone = backbone


    def cont_forward(self, og_x1, og_x2, x1, x2):
        if self.cont_method == 'simsiam':
            f, h = self.cont_projector, self.cont_predictor
            z1, z2 = f(x1), f(x2)
            p1, p2 = h(z1), h(z2)

            # Loss calculation
            cont_loss = (D(p1, z2) / 2) + (D(p2, z1) / 2)

        elif self.cont_method =='simclrv1':
            # Obtains encoded representations of x views
            z1 = self.cont_projector(x1)
            z2 = self.cont_projector(x2)

            # Calculates loss based on those representations
            cont_loss = NT_XentLoss(z1, z2, temperature=self.temp)

        elif self.cont_method == 'cpc':
            combined_x = torch.concat([x1, x2])
            og_x = torch.concat([og_x1, og_x2])

            z, c = self.cpc.get_autoregressor_latent(combined_x)
            cont_loss, accuracy = self.cpc.loss.get(og_x, z, c)

        return cont_loss


    def pred_forward(self, x1, x2, y_augs):
        # We are predicting what augs have bee applied (multi-label)
        if self.pred_method == 'trans':
            full_x = torch.concat([x1, x2])
            full_x = self.predict_projector(full_x)
        
            full_y = y_augs.repeat(2, 1)

            pred_loss, mean_ap = self.pred_loss(full_x, full_y)

        return pred_loss


    def forward(self, packed_data, y_augs):

        og_x1, og_x2 = packed_data
        batch_size = og_x1.shape[0]

        # If we dont use an adapter, we have a single backbone with no extra lin layers or otherwise 
        if self.adapter == 'None':
            x1 = self.backbone(og_x1)
            x2 = self.backbone(og_x2)

            cont_loss = self.cont_forward(og_x1=og_x1,
                og_x2=og_x2,
                x1=x1,
                x2=x2)

            pred_loss = self.pred_forward(x1=x1,
                x2=x2,
                y_augs=y_augs)


        # If using a split net, we have differing linear output layers, but same elsewhere 
        elif self.adapter == 'split':
            x1_cont = self.backbone(og_x1, task_int=0)
            x2_cont = self.backbone(og_x2, task_int=0)

            x1_pred =  self.backbone(og_x1, task_int=1)
            x2_pred = self.backbone(og_x2, task_int=1)

            # If we use adapters and then avg final layers
            if self.backbone.num_outs == 1:
                avg_x1 = (x1_cont + x1_pred) /2
                avg_x2 = (x2_cont + x2_pred) /2

                cont_loss = self.cont_forward(og_x1=og_x1,
                    og_x2=og_x2,
                    x1=avg_x1,
                    x2=avg_x2)

                pred_loss = self.pred_forward(x1=avg_x1,
                    x2=avg_x2,
                    y_augs=y_augs)

            else:
                # If we use adapters and have split output
                cont_loss = self.cont_forward(og_x1=og_x1,
                    og_x2=og_x2,
                    x1=x1_cont,
                    x2=x2_cont)

                pred_loss = self.pred_forward(x1=x1_pred,
                    x2=x2_pred,
                    y_augs=y_augs)


        # If using an adapter, we have task specific params, and can have either a single or split output
        elif self.adapter in ['bn', 'series', 'parallel']:
            x1_cont = self.backbone(og_x1, task_int=0)
            x2_cont = self.backbone(og_x2, task_int=0)

            x1_pred =  self.backbone(og_x1, task_int=1)
            x2_pred = self.backbone(og_x2, task_int=1)

            cont_loss = self.cont_forward(og_x1=og_x1,
                og_x2=og_x2,
                x1=x1_cont,
                x2=x2_cont)

            pred_loss = self.pred_forward(x1=x1_pred,
                x2=x2_pred,
                y_augs=y_augs)

        else:
            raise ValueError(f'Adapter type not recgnised: {self.adapter}')



        # cont_x1 = self.backbone(x1, )
        # cont_x2 = self.backbone(x2, )

        # pred_x1 = self.backbone(x1, )
        # pred_x2 = self.backbone(x2, task_int=)
        # x1 = self.backbone(x1)
        # x2 = self.backbone(x2)

        

        final_loss = self.loss_combination(cont_loss=cont_loss, 
            pred_loss=pred_loss, 
            pred_weight=self.pred_weight)

        return final_loss, pred_loss.item(), cont_loss.item()


