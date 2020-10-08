import torch
from torch.utils.data import DataLoader


class MatchDataLoader(DataLoader):

    def __init__(self, winner_id, loser_id, winner_location, loser_location,
                 winner_score, loser_score, winner_boxscore, loser_boxscore):
        self.winner_id = torch.reshape(torch.tensor(winner_id), (-1, 1))
        self.loser_id = torch.reshape(torch.tensor(loser_id), (-1, 1))
        self.winner_location = torch.reshape(torch.tensor(winner_location), (-1, 1))
        self.loser_location = torch.reshape(torch.tensor(loser_location), (-1, 1))
        self.winner_score = torch.reshape(torch.tensor(winner_score), (-1, 1)).float()
        self.loser_score = torch.reshape(torch.tensor(loser_score), (-1, 1)).float()
        self.winner_boxscore = torch.tensor(winner_boxscore)
        self.loser_boxscore = torch.tensor(loser_boxscore)