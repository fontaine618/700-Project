import torch
import torch.nn as nn
import torch.nn.functional as F


class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()
        self.value = nn.Parameter(torch.tensor(1.))
        self.value.local = False

    def forward(self, mean, variance, x):
        # get values
        m = self.value
        # add
        mean = mean + m * x
        return mean, variance


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
        self.atanh_cor = nn.Parameter(torch.tensor(03.))
        self.mean.local = False
        self.mult.local = False
        self.log_var.local = False
        self.atanh_cor.local = False

    def forward(self, mean0, var0, mean1, var1):
        mean = torch.cat([mean0, mean1], 1)
        mean = mean * self.mult + self.mean
        sig2 = torch.exp(self.log_var)
        cor = torch.tanh(self.atanh_cor)
        var = torch.zeros((var0.size(0), 2, 2))
        var[:, 0, 0] = var0.view(-1,) * self.mult ** 2 + sig2
        var[:, 1, 1] = var1.view(-1,) * self.mult ** 2 + sig2
        var[:, 0, 1] = cor * sig2
        var[:, 1, 0] = cor * sig2
        return mean, var


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, mean, var, dim=1):
        # square
        m = mean ** 2 + var
        m4 = mean ** 4 + 6. * mean ** 2 * var + 3. * var ** 2
        v = m4 + m ** 2
        # sum
        m = torch.sum(m, dim, keepdim=True)
        v = torch.sum(v, dim, keepdim=True)
        return m, v


class Subtract(nn.Module):

    def __init__(self):
        super(Subtract, self).__init__()

    def forward(self, mean0, var0, mean1, var1):
        return mean0 - mean1, var0 + var1


class Sum(nn.Module):

    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, mean0, var0, mean1, var1):
        return mean0 + mean1, var0 + var1


class Difference(nn.Module):

    def __init__(self):
        super(Difference, self).__init__()

    def forward(self, mean, var):
        m = mean[:, 0] - mean[:, 1]
        v = var[:, 0, 0] + var[:, 1, 1] - 2. * var[:, 0, 1]
        return m, v


class LinearMeanVariance(nn.Module):

    _n = 0

    def __init__(self, in_dim, out_dim):
        super(LinearMeanVariance, self).__init__()
        self.mean_map = nn.Linear(in_dim, out_dim, bias=False)
        self.mean_map.weight.data = torch.randn((out_dim, in_dim))
        self.variance_map = nn.Linear(in_dim, out_dim, bias=False)
        self.variance_map.weight.data = torch.zeros((out_dim, in_dim))
        self.mean_map.weight.local = True
        self.variance_map.weight.local = True
        self.mean_map.weight.n = LinearMeanVariance._n
        self.variance_map.weight.n = LinearMeanVariance._n
        LinearMeanVariance._n += 1

    def forward(self, x):
        mean = self.mean_map(x)
        var = self.variance_map(x)
        var = torch.exp(var)
        return mean, var


class Prior(nn.Module):

    def __init__(self, mean=0., var=1.):
        super(Prior, self).__init__()
        self.mean = torch.tensor(mean)
        self.var = torch.tensor(var)

    def forward(self, mean, var):
        v = 1. / (1. / self.var + 1. / var)
        m = v * (mean / var + self.mean / self.var)
        return m, v

class InnerProduct(nn.Module):

    def __init__(self):
        super(InnerProduct, self).__init__()

    def forward(self, off_mean, off_var, def_mean, def_var):
        m = torch.sum(off_mean * def_mean, 1, keepdim=True)
        v = torch.sum(off_mean ** 2 * def_mean ** 2 + off_var * def_mean ** 2 + def_var * off_mean ** 2, 1, keepdim=True)
        return m, v
