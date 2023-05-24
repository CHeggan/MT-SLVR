"""
File contains the necessary classes for SimCLR contrastive learning module
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# NORMALISED TEMPERATURE SCALED CROSS-ENTROPY LOSS
###############################################################################
def NT_XentLoss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)

###############################################################################
# PROJECTION MLP
###############################################################################
class projection_MLP_SimCLR(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

###############################################################################
# SIMCLR V1 FRAMEWORK
###############################################################################
class SimCLR_V1(nn.Module):
    def __init__(self, backbone, temp=0.07, final_out_dim=256):
        """Main SimCLR V1 module. Takes a batch of 2 views of each contained 
            original x, uses temp scaled cross-entropy loss to push views of 
            same x together and push views of different xs apart

        Args:
            backbone (torch nn module ): The backbone encoder to use before simclr module
            final_out_dim (int, optional): The output dimensionality of the projection 
                layer . Defaults to 256.
        """
        super(SimCLR_V1, self).__init__()

        self.temp = temp

        # SimCLR_v1 has only one projection head as opposed to simsiam which 
        #   has projection and prediction
        self.backbone = backbone
        # Projection head is more layers in simclr V2, we only consider v1 here
        self.projector = projection_MLP_SimCLR(in_dim=backbone.output_dim, out_dim=final_out_dim)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )


    def forward(self, x1, x2):
        # Obtains encoded representations of x views
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # Calculates loss based on those representations
        loss = NT_XentLoss(z1, z2, temperature=self.temp)
        return loss

    
