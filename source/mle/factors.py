import torch
import torch.nn as nn


class TeamPlusConf(nn.Module):

    def __init__(self):
        super(TeamPlusConf, self).__init__()
        self.conf_weight = 1. #nn.Parameter(torch.tensor(1.))
        # self.conf_weight.local = False

    def forward(self, team_skill, conf_skill, team, conf):
        return team_skill[team.view(-1), :] + self.conf_weight * conf_skill[conf.view(-1), :]


class InnerProduct(nn.Module):

    def __init__(self):
        super(InnerProduct, self).__init__()

    def forward(self, skill0, skill1):
        return torch.sum(skill0 * skill1, 1, keepdim=True)


class L2Norm(nn.Module):

    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, skill0, skill1):
        return torch.sum((skill0 - skill1) ** 2, 1, keepdim=True)


class HomeField(nn.Module):

    def __init__(self):
        super(HomeField, self).__init__()
        self.advantage = nn.Parameter(torch.tensor(1.))
        self.advantage.local = False

    def forward(self, skill, loc):
        return skill + self.advantage * loc.float()


class Gaussian(nn.Module):

    def __init__(self):
        super(Gaussian, self).__init__()
        self.mean = nn.Parameter(torch.tensor(70.))
        self.mult = nn.Parameter(torch.tensor(10.))
        self.log_var = nn.Parameter(torch.tensor(1.))
        self.atanh_cor = nn.Parameter(torch.tensor(1.))
        self.mean.local = False
        self.mult.local = False
        self.log_var.local = False
        self.atanh_cor.local = False

    def forward(self, skill):
        return skill * self.mult + self.mean

    def likelihood(self, score_pred, score):
        # log det term
        sig2 = torch.exp(self.log_var)
        cor = torch.tanh(self.atanh_cor)
        logdet = torch.log((2. * 3.141592653) ** 2 * sig2 ** 2 * (1. - cor ** 2))
        Sigma_inv = (torch.eye(2) * (1. + cor) - cor) / (sig2 * (1. - cor ** 2))
        # quadratic term
        diff = score_pred - score
        quad_term = torch.sum(torch.matmul(diff, Sigma_inv) * diff)
        # return
        m = score.size(0)
        llk = - 0.5 * (m * logdet + quad_term)
        return llk




