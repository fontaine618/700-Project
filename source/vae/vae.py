import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from source.data.load import MatchDataset
from source.vae.model import VAE

season_results = MatchDataset(2017, "season")
tournament_results = MatchDataset(2017, "tournament")

data = MatchDataset(2017, "season")

data_loader = DataLoader(data, batch_size=500, shuffle=False)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

model = VAE(data.n_teams, data.n_conf, 10)
model.to(DEVICE)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

mse = nn.MSELoss(reduction="sum")

for i in range(100):

    for step in ["E", "M"]:

        for j in range(3):

            total_loss = 0.0
            total_mse = 0.0

            for WTeamID, LTeamID,  WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ in data_loader:
                WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc = \
                    WTeamID.to(DEVICE), LTeamID.to(DEVICE), WTeamConf.to(DEVICE), LTeamConf.to(DEVICE), \
                    WLoc.to(DEVICE), LLoc.to(DEVICE)

                optimizer.zero_grad()

                for name, param in model.named_parameters():
                    if param.local:
                        param.requires_grad = (step == "E")

                score_mean, score_var = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)

                score = torch.cat([WScore, LScore], 1).to(DEVICE)
                loss = model.loss(score_mean, score_var, score) * 0.5 / data.n_matches
                loss.backward()

                # with torch.no_grad():
                #     for name, param in model.named_parameters():
                #         if param.requires_grad:
                #             param -= 0.1 * param.grad

                optimizer.step()

                total_loss += loss.item()

                with torch.no_grad():
                    total_mse += mse(score_mean, score).item() * 0.5 / data.n_matches

            print("{:10} {:5} {:5} {:10.6f} {:10.6f}".format(i, step, j, total_loss, total_mse))

for name, param in model.named_parameters():
    if not param.local:
        print(name, param)

L = model.decoder.gaussian.L

torch.matmul(L, L.transpose(0, 1))
