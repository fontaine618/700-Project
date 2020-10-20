from abc import ABC

import torch
import torch.nn as nn


def _nearest_orthogonal(Z):
    u, _, v = torch.svd(Z)
    return torch.mm(u, v.t())


class OffDefTeamConfLVM(nn.Module, ABC):

    def __init__(self, n_team, n_conf, dim, model):
        super(OffDefTeamConfLVM, self).__init__()
        self.offense_team = nn.Parameter(torch.randn((n_team, dim)))
        self.defense_team = nn.Parameter(torch.randn((n_team, dim)))
        self.offense_conf = nn.Parameter(torch.randn((n_conf, dim)))
        self.defense_conf = nn.Parameter(torch.randn((n_conf, dim)))
        self.model = model

    def forward(self, *args):
        return self.model(self.offense_team, self.defense_team, self.offense_conf, self.defense_conf, *args)

    def project(self):
        with torch.no_grad():
            self.offense_team.data = _nearest_orthogonal(self.offense_team)
            self.defense_team.data = _nearest_orthogonal(self.defense_team)
            self.offense_conf.data = _nearest_orthogonal(self.offense_conf)
            self.defense_conf.data = _nearest_orthogonal(self.defense_conf)

