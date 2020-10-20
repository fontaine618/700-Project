import torch
import torch.nn as nn
from source.mle.factors import TeamPlusConf, InnerProduct, HomeField, Gaussian, L2Norm


def _nearest_orthogonal(Z):
    u, _, v = torch.svd(Z)
    return torch.mm(u, v.t())


class ScoreModel(nn.Module):

    def __init__(self, n_team, n_conf, dim, model="InnerProduct"):
        super(ScoreModel, self).__init__()
        self.n_team = n_team
        self.n_conf = n_conf
        self.dim = dim
        # latent variables
        self.team_off = nn.Parameter(torch.randn((n_team, dim)))
        self.team_def = nn.Parameter(torch.randn((n_team, dim)))
        self.conf_off = nn.Parameter(torch.randn((n_conf, dim)))
        self.conf_def = nn.Parameter(torch.randn((n_conf, dim)))
        self.project()
        self.team_off.local = True
        self.team_def.local = True
        self.conf_off.local = True
        self.conf_def.local = True
        # factors
        self.team_plus_conf = TeamPlusConf()
        self.model = InnerProduct() if model == "InnerProduct" else L2Norm()
        self.home_field = HomeField()
        self.gaussian = Gaussian()

    def forward(self,
                win_team, los_team,
                win_conf, los_conf,
                win_loc, los_loc
                ):
        # get per match skills
        win_off = self.team_plus_conf(self.team_off, self.conf_off, win_team, win_conf)
        win_def = self.team_plus_conf(self.team_def, self.conf_def, win_team, win_conf)
        los_off = self.team_plus_conf(self.team_off, self.conf_off, los_team, los_conf)
        los_def = self.team_plus_conf(self.team_def, self.conf_def, los_team, los_conf)
        # get per team skills
        win = self.model(win_off, los_def)
        los = self.model(los_off, win_def)
        # add home-field
        win = self.home_field(win, win_loc)
        los = self.home_field(los, los_loc)
        # get predicted scores
        skill = torch.cat([win, los], 1)
        score = self.gaussian(skill)
        return score

    def project(self):
        with torch.no_grad():
            self.team_off.data = _nearest_orthogonal(self.team_off)
            self.team_def.data = _nearest_orthogonal(self.team_def)
            self.conf_off.data = _nearest_orthogonal(self.conf_off)
            self.conf_def.data = _nearest_orthogonal(self.conf_def)

    def loss(self, score_pred, score):
        return - self.gaussian.likelihood(score_pred, score)

    def evaluate(self, score_pred, score):
        llk = self.gaussian.likelihood(score_pred, score).item() / score.size(0)
        mse = nn.MSELoss(reduction="sum")(score_pred, score).item() * 0.5 / score.size(0)
        pred_result = score_pred[:, 0] - score_pred[:, 1] >= 0
        acc = torch.mean(pred_result.float()).item()

        sig2 = torch.exp(self.gaussian.log_var)
        cor = torch.tanh(self.gaussian.atanh_cor)
        mean = score_pred[:, 0] - score_pred[:, 1]
        sd = torch.sqrt(2. * sig2 * (1. - cor))
        proba = torch.distributions.Normal(0., 1.).cdf(mean / sd)

        cross_entropy = - torch.mean(torch.log(proba))
        return{
            "llk": llk, "mse": mse, "acc": acc, "cross_entropy": cross_entropy
        }

    def predict_conference(self):
        return self.conf_off, self.conf_def

    def predict_teams(self):
        return self.team_off, self.team_def

    def predict_team_plus_conference(self, conf_id):
        team_off = self.team_off + self.conf_off[conf_id, :]
        team_def = self.team_def + self.conf_def[conf_id, :]
        return team_off, team_def

