from abc import ABC

import torch
import torch.nn as nn
from source.vae.encoder import Encoder
from source.vae.decoder import ScoreModel


class VAE(nn.Module, ABC):

    def __init__(self, n_team, n_conf, n_dim):
        super(VAE, self).__init__()
        self.n_team = n_team
        self.n_conf = n_conf
        self.n_dim = n_dim
        self.encoder_team_off = Encoder(n_team, n_dim)
        self.encoder_team_def = Encoder(n_team, n_dim)
        self.encoder_conf_off = Encoder(n_conf, n_dim)
        self.encoder_conf_def = Encoder(n_conf, n_dim)
        self.decoder = ScoreModel()

    def initialize(self, team_off, team_def, conf_off, conf_def):
        self.encoder_team_off.linear.mean_map.data = team_off
        self.encoder_team_def.linear.mean_map.data = team_def
        self.encoder_conf_off.linear.mean_map.data = conf_off
        self.encoder_conf_def.linear.mean_map.data = conf_def
        self.encoder_team_off.linear.log_var_map.data = torch.ones_like(team_off) * -1.
        self.encoder_team_def.linear.log_var_map.data = torch.ones_like(team_def) * -1.
        self.encoder_conf_off.linear.log_var_map.data = torch.ones_like(conf_off) * -1.
        self.encoder_conf_def.linear.log_var_map.data = torch.ones_like(conf_def) * -1.

    def forward(self,
                win_team_id, los_team_id,
                win_conf_id, los_conf_id,
                win_loc, los_loc
                ):
        # One-hot encoding
        win_team_id = torch.nn.functional.one_hot(win_team_id.view(-1), self.n_team).float()
        los_team_id = torch.nn.functional.one_hot(los_team_id.view(-1), self.n_team).float()
        win_conf_id = torch.nn.functional.one_hot(win_conf_id.view(-1), self.n_conf).float()
        los_conf_id = torch.nn.functional.one_hot(los_conf_id.view(-1), self.n_conf).float()
        # Encoder
        win_team_off = self.encoder_team_off(win_team_id)
        win_team_def = self.encoder_team_def(win_team_id)
        los_team_off = self.encoder_team_off(los_team_id)
        los_team_def = self.encoder_team_def(los_team_id)
        win_conf_off = self.encoder_conf_off(win_conf_id)
        win_conf_def = self.encoder_conf_def(win_conf_id)
        los_conf_off = self.encoder_conf_off(los_conf_id)
        los_conf_def = self.encoder_conf_def(los_conf_id)
        # Decoder
        score_mean, score_var, _, _ = self.decoder(
            *win_team_off,
            *win_conf_off,
            *los_team_def,
            *los_conf_def,
            *los_team_off,
            *los_conf_off,
            *win_team_def,
            *win_conf_def,
            win_loc,
            los_loc
        )
        return score_mean, score_var

    def loss(self, score_mean, score_var, score):
        # log det term
        sig2 = torch.exp(self.decoder.gaussian.log_var)
        cor = torch.tanh(self.decoder.gaussian.atanh_cor)
        logdet = torch.log((2. * 3.141592653) ** 2 * sig2 ** 2 * (1. - cor ** 2))
        Sigma_inv = (torch.eye(2) * (1. + cor) - cor) / (sig2 * (1. - cor ** 2))
        # quad term
        diff = score_mean - score
        quad_term = torch.sum(torch.matmul(diff, Sigma_inv) * diff)
        # trace term
        trace = torch.sum(score_var) / (sig2 * (1. - cor ** 2))
        # likelihood term
        m = score.size(0)
        llk = - 0.5 * (m * logdet + trace + quad_term)

        # prior term
        prior = 0.
        for encoder in [
            self.encoder_team_off,
            self.encoder_team_def,
            self.encoder_conf_off,
            self.encoder_conf_def
        ]:
            prior += encoder.elbo()

        # entropy term
        entropy = 0.
        # for encoder in [
        #     self.encoder_team_off,
        #     self.encoder_team_def,
        #     self.encoder_conf_off,
        #     self.encoder_conf_def
        # ]:
        #     entropy += encoder.entropy()
        entropy_score = 0.5 * (torch.log(torch.tensor(2. * 3.141592653)) + torch.logdet(score_var) + 1.)
        entropy += torch.sum(entropy_score)

        # print(llk.item(), prior.item(), entropy.item())
        loss = -(llk + prior + entropy)


        # # MSE
        # mse = nn.MSELoss(reduction="sum")
        # loss = mse(score_mean, score) * 0.5 + torch.sum(score_var[:, 0, 0]) + torch.sum(score_var[:, 1, 1])
        return loss
