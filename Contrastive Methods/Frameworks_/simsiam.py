"""
File contains the necessary classes for SimSiam contrastive learning module
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# MAIN FUNCTIONALITY OF SIMSIAM
###############################################################################
def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

###############################################################################
# PROJECTION & PREDICTION HEAD CLASSES
###############################################################################
class projection_MLP(nn.Module):
    def __init__(self, in_dim, num_layers=3, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = num_layers

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

###############################################################################
# MAIN SIMCLR
###############################################################################
class SimSiam(nn.Module):
    def __init__(self, backbone, connection_dim=2048, final_out_dim=2048):
        """Main simsiam module, takes a batch of 2 views of each contained 
            original x and uses a stop grad based loss to push same x views 
            together. Does not use negative sampling

        Args:
            backbone (torch nn module): The backbone encoder to use before 
                simsiam module
            connection_dim (int, optional): The dimensionality of the connection 
                between projection and prediction modules. Defaults to 2048.
            final_out_dim (int, optional): The final output dimensionality of the 
                simsiam module. Defaults to 2048.
        """
        super(SimSiam, self).__init__()

        # When inferencing, only the backbone with its fc layer is used, we
        #   write in an explicit function for this for convenience when testing
        self.backbone = backbone
        # Projection layer can be 2 or 3 layers, 3rd layer still created but not updated
        self.projector = projection_MLP(backbone.output_dim,
            num_layers=3, hidden_dim=2048, out_dim=connection_dim)

        # Encoder is made up of the backbone (e.g. resnet with fc output layer)
        #   and a projection layer
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        # Predictor is the final layer of simsiam and is only actually used 
        #   gradient wise for one side of the siamese net
        self.predictor = prediction_MLP(in_dim=connection_dim, hidden_dim=512, 
            out_dim=final_out_dim)
    
    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)

        # Loss calculation
        L = (D(p1, z2) / 2) + (D(p2, z1) / 2)
        return L

    def inference_forward(self, x):
        return self.encoder(x)


###############################################################################
# TESTING
###############################################################################
# encoder = resnet_selection(1, 512, 'resnet18')

# device = torch.device('cuda')
# encoder = encoder.to(device)

# # simsiam = SimSiam(encoder, 512)
# # print(simsiam)
# # print(count_parameters(simsiam))


# x1 = torch.rand(10, 1, 160000).to(device)
# # x2 = torch.rand(10, 1, 160000).to(device)

# out = encoder.forward(x1)
# print(out)