import torch
import torch.nn as nn
from source.vae.encoder import Encoder
from source.vae.decoder import ScoreModel


class VAE(nn.Module):

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
        # logdet = torch.logdet(2 * 3.141592654 * score_var)
        # inv = torch.inverse(score_var)
        # d = score_mean - score
        # z = torch.einsum("ij, ijk, ik->i", d, inv, d)
        #
        # loss = -0.5 * (logdet + z)
        #
        # return torch.sum(loss)

        mse = nn.MSELoss(reduction="sum")
        return mse(score_mean, score) # + torch.sum(score_var[:, 0, 0] + score_var[:, 1, 1])