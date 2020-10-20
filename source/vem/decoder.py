import torch
import torch.nn as nn
from source.vem.factors import Gaussian
from source.vem.factors import InnerProduct
from source.vem.factors import Sum, Add


class MeanModel(nn.Module):

    def __init__(self):
        super(MeanModel, self).__init__()
        self.sum = Sum()
        self.inner_product = InnerProduct()

    def forward(self,
                team_off,
                conf_off,
                team_def,
                conf_def,
                ):
        off = self.sum(team_off, conf_off)
        deff = self.sum(team_def, conf_def)
        mean = self.inner_product(off, deff)
        return mean


class ScoreModel(nn.Module):

    def __init__(self):
        super(ScoreModel, self).__init__()
        self.mean_model = MeanModel()
        self.gaussian = Gaussian()
        self.home_field = Add()

    def forward(self,
                win_team_off,
                win_conf_off,
                los_team_def,
                los_conf_def,
                los_team_off,
                los_conf_off,
                win_team_def,
                win_conf_def,
                win_loc, los_loc
            ):
        # winning mean
        win = self.mean_model(
                win_team_off,
                win_conf_off,
                los_team_def,
                los_conf_def
        )
        win = self.home_field(win, win_loc)
        # losing mean
        los = self.mean_model(
                los_team_off,
                los_conf_off,
                win_team_def,
                win_conf_def
        )
        los = self.home_field(los, los_loc)
        # scores and differences
        score = self.gaussian(win, los)

        return score

    def llk(self, x, score):
        return self.gaussian.llk(x, score)