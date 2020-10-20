from abc import ABC

import torch
import torch.nn as nn
from source.vem.encoder import Encoder
from source.vem.decoder import ScoreModel


class VariationalEM(nn.Module, ABC):

    def __init__(self, n_team, n_conf, n_dim):
        super(VariationalEM, self).__init__()
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
        self.encoder_team_off.linear.log_var_map.data = torch.ones_like(team_off) * -3.
        self.encoder_team_def.linear.log_var_map.data = torch.ones_like(team_def) * -3.
        self.encoder_conf_off.linear.log_var_map.data = torch.ones_like(conf_off) * -3.
        self.encoder_conf_def.linear.log_var_map.data = torch.ones_like(conf_def) * -3.

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
        score = self.decoder(
            win_team_off,
            win_conf_off,
            los_team_def,
            los_conf_def,
            los_team_off,
            los_conf_off,
            win_team_def,
            win_conf_def,
            win_loc,
            los_loc
        )
        return score

    def loss(self, sample, score):
        """
        ELBO(Q) = E_Q{log P(V|H)} - KL(Q||Prior)
        where
        KL(Q||Prior) = 1/2 sum 1 + log sig2 - mu2 - sig2
        and
        E_Q{log P(V|H)} ~ sample average of log P(V|H)
        """
        mc_exp_llk = self.decoder.llk(sample, score)
        kl_q_prior = 0.
        for encoder in [
            self.encoder_team_off,
            self.encoder_team_def,
            self.encoder_conf_off,
            self.encoder_conf_def
        ]:
            kl_q_prior += encoder.kl()
        return - mc_exp_llk - kl_q_prior

    def evaluate(self, score_pred, score):
        llk = self.decoder.llk(score_pred, score).item() / score.size(0)
        mse = nn.MSELoss(reduction="sum")(score_pred, score).item() * 0.5 / score.size(0)
        pred_result = score_pred[:, 0] - score_pred[:, 1] >= 0
        acc = torch.mean(pred_result.float()).item()

        sig2 = torch.exp(self.decoder.gaussian.log_var)
        cor = torch.tanh(self.decoder.gaussian.atanh_cor)
        mean = score_pred[:, 0] - score_pred[:, 1]
        sd = torch.sqrt(2. * sig2 * (1. - cor))
        proba = torch.distributions.Normal(0., 1.).cdf(mean / sd).clamp(1.e-6, 1. - 1.e-6)

        cross_entropy = - torch.mean(torch.log(proba)).item()
        return llk, mse, acc, cross_entropy

    def evaluate_average(self, data, n_sample=100):
        llk = 0.
        mse = 0.
        acc = 0.
        cross_entropy = 0.

        for _ in range(n_sample):

            WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ = data[:]
            with torch.no_grad():
                score = torch.cat([WScore, LScore], 1)
                score_pred = self.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)
                metrics = self.evaluate(score_pred, score)

            llk += metrics[0]
            mse += metrics[1]
            acc += metrics[2]
            cross_entropy += metrics[3]

        return{
            "llk": llk / n_sample, "mse": mse / n_sample,
            "acc": acc / n_sample, "cross_entropy": cross_entropy / n_sample
        }


