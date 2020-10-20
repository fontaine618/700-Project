import pandas as pd
import torch
from torch.utils.data import Dataset


PATH = "./data/mens-machine-learning-competition-2018/DataFiles/"


class MatchDataset(Dataset):

    def __init__(self, year, type="season", path=PATH):
        if type == "tournament":
            file = "NCAATourneyDetailedResults.csv"
        else:  # base case is the season
            file = "RegularSeasonDetailedResults.csv"
        # load all season results
        results = pd.read_csv(filepath_or_buffer=path + file)
        # subset to season
        season_results = results[results["Season"] == year]
        keep = season_results["NumOT"] == 0
        season_results = season_results.loc[keep]
        self.n_matches = len(season_results)
        # conferences
        results = pd.read_csv(filepath_or_buffer=path + "TeamConferences.csv")
        self.conference_names = pd.read_csv(filepath_or_buffer=path + "Conferences.csv")
        self.conference_names.set_index("ConfAbbrev", inplace=True)
        # subset to season
        conferences = results[results["Season"] == year].drop(columns="Season")
        conferences.set_index("TeamID", inplace=True)
        self.team_conf_dict = conferences
        self.conferences = pd.Categorical(conferences["ConfAbbrev"]).categories
        self.n_conf = len(self.conferences)
        conferences["ConfID"] = pd.Categorical(conferences["ConfAbbrev"]).codes
        # add to df
        season_results["WTeamConf"] = conferences.loc[season_results["WTeamID"]]["ConfID"].values
        season_results["LTeamConf"] = conferences.loc[season_results["LTeamID"]]["ConfID"].values
        # teams
        teams = pd.read_csv(filepath_or_buffer=path + "Teams.csv", index_col="TeamID")
        team_ids = pd.concat([season_results["WTeamID"], season_results["LTeamID"]])
        self.teams = pd.Categorical(team_ids).categories
        self.team_names = teams.loc[self.teams, "TeamName"]
        self.n_teams = len(self.teams)
        # store values
        self.index = torch.tensor(season_results.index)
        self.WTeamID = torch.tensor(season_results["WTeamID"].replace(self.teams, range(self.n_teams)).values).reshape((-1, 1))
        self.LTeamID = torch.tensor(season_results["LTeamID"].replace(self.teams, range(self.n_teams)).values).reshape((-1, 1))
        self.WTeamConf = torch.tensor(season_results["WTeamConf"].values, dtype=torch.int64).reshape((-1, 1))
        self.LTeamConf = torch.tensor(season_results["LTeamConf"].values, dtype=torch.int64).reshape((-1, 1))
        self.WLoc = torch.tensor(season_results["WLoc"].replace({"H": 1, "N": 0, "A": -1}).values).reshape((-1, 1)).float()
        self.LLoc = torch.tensor(season_results["WLoc"].replace({"A": 1, "N": 0, "H": -1}).values).reshape((-1, 1)).float()
        self.WScore = torch.tensor(season_results["WScore"].values).reshape((-1, 1)).float()
        self.LScore = torch.tensor(season_results["LScore"].values).reshape((-1, 1)).float()
        self.WBox = torch.tensor(season_results[['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
                                    'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].values)
        self.LBox = torch.tensor(season_results[['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR',
                                    'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].values)
        # some dimensions

    def __len__(self):
        return len(self.WTeamID)

    def __getitem__(self, i):
        return (
            self.WTeamID[i], self.LTeamID[i],
            self.WTeamConf[i], self.LTeamConf[i],
            self.WLoc[i], self.LLoc[i],
            self.WScore[i], self.LScore[i],
            self.WBox[i, :], self.LBox[i, :]
        )

