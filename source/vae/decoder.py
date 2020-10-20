import torch
import torch.nn as nn
from source.vae.factors import Gaussian
from source.vae.factors import L2Norm, InnerProduct
from source.vae.factors import Subtract, Sum, Difference, Add


class MeanModel(nn.Module):

    def __init__(self):
        super(MeanModel, self).__init__()
        self.sum = Sum()
        # self.difference = Subtract()
        # self.l2norm = L2Norm()
        self.inner_product = InnerProduct()

    def forward(self,
                team_off_mean, team_off_var,
                conf_off_mean, conf_off_var,
                team_def_mean, team_def_var,
                conf_def_mean, conf_def_var,
                ):
        off_mean, off_var = self.sum(team_off_mean, team_off_var, conf_off_mean, conf_off_var)
        def_mean, def_var = self.sum(team_def_mean, team_def_var, conf_def_mean, conf_def_var)
        # diff_mean, diff_var = self.difference(off_mean, off_var, def_mean, def_var)
        # mean, var = self.l2norm(diff_mean, diff_var)
        mean, var = self.inner_product(off_mean, off_var, def_mean, def_var)
        return mean, var


class ScoreModel(nn.Module):

    def __init__(self):
        super(ScoreModel, self).__init__()
        self.mean_model = MeanModel()
        self.gaussian = Gaussian()
        self.difference = Difference()
        self.home_field = Add()

    def forward(self,
                win_team_off_mean, win_team_off_var,
                win_conf_off_mean, win_conf_off_var,
                los_team_def_mean, los_team_def_var,
                los_conf_def_mean, los_conf_def_var,
                los_team_off_mean, los_team_off_var,
                los_conf_off_mean, los_conf_off_var,
                win_team_def_mean, win_team_def_var,
                win_conf_def_mean, win_conf_def_var,
                win_loc, los_loc
            ):
        # winning mean
        win_mean, win_var = self.mean_model(
                win_team_off_mean, win_team_off_var,
                win_conf_off_mean, win_conf_off_var,
                los_team_def_mean, los_team_def_var,
                los_conf_def_mean, los_conf_def_var
        )
        win_mean, win_var = self.home_field(win_mean, win_var, win_loc)
        # losing mean
        los_mean, los_var = self.mean_model(
                los_team_off_mean, los_team_off_var,
                los_conf_off_mean, los_conf_off_var,
                win_team_def_mean, win_team_def_var,
                win_conf_def_mean, win_conf_def_var
        )
        los_mean, los_var = self.home_field(los_mean, los_var, los_loc)
        # scores and differences
        score_mean, score_var = self.gaussian(win_mean, win_var, los_mean, los_var)

        # print(score_mean[0, :], score_var[0, :, :])

        diff_mean, diff_var = self.difference(score_mean, score_var)

        return score_mean, score_var, diff_mean, diff_var