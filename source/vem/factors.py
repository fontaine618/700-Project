import torch
import torch.nn as nn


class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()
        self.value = nn.Parameter(torch.tensor(1.))
        self.value.local = False

    def forward(self, x, where):
        return x + self.value * where


class Gaussian(nn.Module):
    """
    O|S = mu + S + e, e~N(0, Sigma)
    O ~ N(mu + mu_S, Sigma_M + Sigma)
    with
    mu = (mu, mu)'
    Sigma = (1-rho) sigma^2 I_2 + rho 11'
    """

    def __init__(self):
        super(Gaussian, self).__init__()
        self.mean = nn.Parameter(torch.tensor(70.))
        self.mult = nn.Parameter(torch.tensor(100.))
        self.log_var = nn.Parameter(torch.tensor(5.))
        self.atanh_cor = nn.Parameter(torch.tensor(0.3))
        self.mean.local = False
        self.mult.local = False
        self.log_var.local = False
        self.atanh_cor.local = False

    def forward(self, x0, x1):
        score = torch.cat([x0, x1], 1) * self.mult * self.mean

        # sample
        # sig = torch.exp(self.log_var * 0.5)
        # cor = torch.tanh(self.atanh_cor)
        # e0 = torch.randn_like(x0)
        # e1 = torch.randn_like(x1)
        # s0 = x0 * self.mult + self.mean + sig * e0
        # s1 = x1 * self.mult + self.mean + sig * (cor * e0 + torch.sqrt(1. - cor**2) * e1)
        #
        # score = torch.cat([s0, s1], 1)

        return score

    def llk(self, x, score):
        # log det term
        sig2 = torch.exp(self.log_var)
        cor = torch.tanh(self.atanh_cor)
        logdet = torch.log((2. * 3.141592653) ** 2 * sig2 ** 2 * (1. - cor ** 2))
        Sigma_inv = (torch.eye(2) * (1. + cor) - cor) / (sig2 * (1. - cor ** 2))
        # quad term
        diff = x - score
        quad_term = torch.sum(torch.matmul(diff, Sigma_inv) * diff)
        # likelihood term
        m = score.size(0)
        return - 0.5 * (m * logdet + quad_term)


class Sum(nn.Module):

    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x0, x1):
        return x0 + x1


class LinearMeanVariance(nn.Module):

    _n = 0

    def __init__(self, in_dim, out_dim):
        super(LinearMeanVariance, self).__init__()
        self.mean_map = nn.Linear(in_dim, out_dim, bias=False)
        self.mean_map.weight.data = torch.randn((out_dim, in_dim))
        self.log_var_map = nn.Linear(in_dim, out_dim, bias=False)
        self.log_var_map.weight.data = torch.ones((out_dim, in_dim)) * -3
        self.mean_map.weight.local = True
        self.log_var_map.weight.local = True
        self.mean_map.weight.n = LinearMeanVariance._n
        self.log_var_map.weight.n = LinearMeanVariance._n
        LinearMeanVariance._n += 1

    def forward(self, x):
        mean = self.mean_map(x)
        log_var = self.log_var_map(x)
        return mean, log_var


class InnerProduct(nn.Module):

    def __init__(self):
        super(InnerProduct, self).__init__()

    def forward(self, x0, x1):
        return torch.sum(x0 * x1, 1, keepdim=True)
