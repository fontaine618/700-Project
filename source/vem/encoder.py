import torch
import torch.nn as nn
from source.vem.factors import LinearMeanVariance


class Encoder(nn.Module):

    def __init__(self, in_dim, out_dim, prior_mean=0., prior_var=1.):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = LinearMeanVariance(in_dim, out_dim)
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    def forward(self, x):
        mean, log_var = self.linear(x)
        return mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)

    def kl(self):
        """
        KL(Q||Prior) = 1/2 sum 1 + log sig2 - mu2 - sig2
        """
        x = torch.eye(self.in_dim)
        mean, log_var = self.linear(x)
        kl = 1. + log_var - mean ** 2 - torch.exp(log_var)
        return torch.sum(kl) * 0.5

