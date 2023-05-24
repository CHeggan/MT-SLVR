"""
File contains the necessary classes for augmentation prediction learning
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics

###############################################################################
# MEAN AVERAGE PRECISION
###############################################################################
def mAP(y_true, y_logits, average_type='macro'):
    return metrics.average_precision_score(y_true, y_logits, average=average_type)

###############################################################################
# AUG PREDICT LOSS W/ MAP FOR MULTI-LABEL
###############################################################################
def multi_label_bce(raw_logits, y):
    # We start with raw logits directly from model
    # We use BCE w/ logits for numerical stability
    loss = F.binary_cross_entropy_with_logits(raw_logits, y)

    # Convert raw logits into probabilities using sigmoid
    y_probs = torch.sigmoid(raw_logits)
    # Calculate binary accuracy 
    #acc = binary_accuracy_w_probs(y, y_probs)
    acc = mAP(y.detach().cpu().numpy(), raw_logits.detach().cpu().numpy())
    return loss, acc

###############################################################################
# PREDICTION HEAD
###############################################################################
class prediction_MLP(nn.Module):
    def __init__(self, in_dim, num_layers=4, hidden_dims=[256, 64, 16], out_dim=2048):
        """Projection head for aug prediction

        Args:
            in_dim (int): _description_
            num_layers (int, optional): Number of layers between backbone and final output. 
                Defaults to 4.
            hidden_dims (list, optional): The dimensions of the hidden layers. 
                Defaults to [256, 64, 16].
            out_dim (int, optional): The final output dimensionality. Defaults to 2048.
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], out_dim)
        )

        self.num_layers = num_layers

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 4:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x 

###############################################################################
# MAIN PREDICTION MODULE
###############################################################################
class PredictAug(nn.Module):
    def __init__(self, backbone, out_dim=1):
        """Main aug prediction module. Takes augmented views x and generate a
            binary output. 

        Args:
            backbone (torch nn module): The backbone encoder to use before 
                aug prediction module
            out_dim (int, optional): The final output dimensionality of the 
                module. Defaults to 1.
        """
        super(PredictAug, self).__init__()

        self.loss_acc_fn = multi_label_bce

        # When inferencing, only the backbone with its fc layer is used, we
        #   write in an explicit function for this for convenience when testing
        self.backbone = backbone

        # Projection module is 4 layers
        self.projector = prediction_MLP(backbone.output_dim, num_layers=4,
            hidden_dims=[256, 64, 16], out_dim=out_dim)

        # Encoder is made up of the backbone (e.g. resnet with fc output layer)
        #   and a projection layer
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )


    def forward(self, x, y):
        x = self.encoder(x)

        loss, mean_ap = self.loss_acc_fn(x, y)

        return loss, mean_ap
        

