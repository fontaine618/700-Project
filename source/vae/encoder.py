import torch
import torch.nn as nn
from source.vae.factors import LinearMeanVariance, Prior


class Encoder(nn.Module):

    def __init__(self, in_dim, out_dim, prior_mean=0., prior_var=1.):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = LinearMeanVariance(in_dim, out_dim)
        self.prior = Prior(prior_mean, prior_var)

    def forward(self, x):
        # return self.linear(x)
        mean, var = self.linear(x)
        return self.prior(mean, var)

    def elbo(self):
        x = torch.eye(self.in_dim)
        mean, var = self.forward(x)
        prior_mean = self.prior.mean
        prior_var = self.prior.var
        elbo = torch.log(2. * 3.141592653 * prior_var) + (var + (mean - prior_mean) ** 2) / prior_var
        return torch.sum(elbo) * - 0.5

    def entropy(self):
        x = torch.eye(self.in_dim)
        _, var = self.forward(x)
        entropy = 0.5 * (torch.log(2. * 3.141592653 * var) + 1.)
        return torch.sum(entropy)