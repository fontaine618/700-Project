import torch
import torch.nn as nn
from source.models.factors import SumAlong


class ScoreModel(nn.Module):

    def __init__(self, model):
        super(ScoreModel, self).__init__()
        self.sum_along = SumAlong()
        self.model = model
        self.affine = torch.nn.Linear(1, 1)
        self.home_field = torch.nn.Linear(1, 1, False)

    def forward(self,
                team_offense, team_defense,
                conf_offense, conf_defense,
                winner_team_id, loser_team_id,
                winner_conf_id, loser_conf_id,
                winner_location, loser_location
                ):
        winner_offense = self.sum_along(team_offense, conf_offense, winner_team_id, winner_conf_id)
        winner_defense = self.sum_along(team_defense, conf_defense, winner_team_id, winner_conf_id)
        loser_offense = self.sum_along(team_offense, conf_offense, loser_team_id, loser_conf_id)
        loser_defense = self.sum_along(team_defense, conf_defense, loser_team_id, loser_conf_id)
        winner_skill = self.model(winner_offense, loser_defense)
        loser_skill = self.model(loser_offense, winner_defense)
        winner_score = self.affine(winner_skill) + self.home_field(winner_location)
        loser_score = self.affine(loser_skill) + self.home_field(loser_location)
        return winner_score, loser_score