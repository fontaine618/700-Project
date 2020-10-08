import torch
import torch.nn as nn


class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()
        self.mean = nn.Parameter(torch.tensor(1.))
        self.log_variance = nn.Parameter(torch.tensor(0.))
        self.mean.local = False
        self.log_variance.local = False

    def forward(self, mean, variance, x):
        # get values
        m = self.mean
        v = torch.exp(self.log_variance)
        # add
        mean = mean + m * x
        variance = variance + v * (x ** 2)
        return mean, variance


class Gaussian(nn.Module):

    def __init__(self):
        super(Gaussian, self).__init__()
        self.mean = nn.Parameter(torch.tensor(0.))
        self.L = nn.Parameter(torch.eye(2) * 1.)
        self.mean.local = False
        self.L.local = False
        # fix L[0, 1]=0 so lower triangular

    def forward(self, mean0, var0, mean1, var1):
        mean = torch.cat([mean0, mean1], 1)
        mean = torch.matmul(mean, self.L) + self.mean
        var = torch.zeros((var0.size(0), 2, 2))
        var0 = var0.view(-1,)
        var1 = var1.view(-1,)
        var[:, 0, 0] = self.L[0, 0] ** 2 * var0
        var[:, 0, 1] = self.L[0, 0] * self.L[1, 0] * var0
        var[:, 1, 0] = self.L[0, 0] * self.L[1, 0] * var0
        var[:, 1, 1] = self.L[1, 1] ** 2 * var1 + self.L[1, 0] ** 2 * var0
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

    def __init__(self, in_dim, out_dim):
        super(LinearMeanVariance, self).__init__()
        self.mean_map = nn.Linear(in_dim, out_dim, bias=False)
        self.variance_map = nn.Linear(in_dim, out_dim, bias=False)
        self.mean_map.weight.local = True
        self.variance_map.weight.local = True

    def forward(self, x):
        mean = self.mean_map(x)
        var = self.variance_map(x)
        var = torch.exp(var)
        return mean, var


class Prior(nn.Module):

    def __init__(self, mean=0., var=1.):
        super(Prior, self).__init__()
        self.mean = mean
        self.var = var

    def forward(self, mean, var):
        v = 1. / (1. / self.var + 1. / var)
        m = v * (mean / var + self.mean / self.var)
        return m, v
