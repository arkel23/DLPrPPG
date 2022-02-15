import torch.nn as nn
import torch
class exp_loss(nn.Module):
    def __init__(self, beta):
        super(exp_loss, self).__init__()
        self.beta = beta

    def forward(self, input, target, exp_peaks):
        return nn.L1Loss()(
            input*(1+self.beta*exp_peaks), target*(1+self.beta*exp_peaks))