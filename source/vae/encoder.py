import torch
import torch.nn as nn
from source.vae.factors import LinearMeanVariance, Prior


class Encoder(nn.Module):

    def __init__(self, in_dim, out_dim, prior_mean=0., prior_var = 1.):
        super(Encoder, self).__init__()
        self.linear = LinearMeanVariance(in_dim, out_dim)
        self.prior = Prior(prior_mean, prior_var)

    def forward(self, x):
        mean, var = self.linear(x)
        return self.prior(mean, var)