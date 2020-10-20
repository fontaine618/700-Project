import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from source.data.load import MatchDataset
from source.vae.model import VAE

season_results = MatchDataset(2017, "season")
tournament_results = MatchDataset(2017, "tournament")

data = MatchDataset(2017, "season")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

K = 2

# # INITIALIZATION
#
# data_loader = DataLoader(data, batch_size=500, shuffle=False)
#
# from source.models.latent_variable_model import OffDefTeamConfLVM
# from source.models.score_model import ScoreModel
# from source.models.factors import Distance
#
# space_model = Distance()
#
# mle = OffDefTeamConfLVM(data.n_teams, data.n_conf, K, ScoreModel(space_model)).to(DEVICE)
# mle.train()
# mse = nn.MSELoss(reduction="sum")
#
# optimizer = torch.optim.SGD(mle.parameters(), lr=0.01, momentum=0.5)
#
#
# for i in range(200):
#
#     total_loss = 0.0
#
#     for WTeamID, LTeamID,  WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ in data_loader:
#         WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore = \
#             WTeamID.to(DEVICE), LTeamID.to(DEVICE), WTeamConf.to(DEVICE), LTeamConf.to(DEVICE), \
#             WLoc.to(DEVICE), LLoc.to(DEVICE), WScore.to(DEVICE), LScore.to(DEVICE)
#
#         optimizer.zero_grad()
#
#         WMean, LMean = mle.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)
#         loss = (mse(WMean, WScore) + mse(LMean, LScore)) * 0.5 / data.WLoc.size(0)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#         with torch.no_grad():
#             mle.project()
#
#     print("{:10} {:10.3f}".format(i, total_loss))
#
#
# team_off = mle.offense_team.transpose(0, 1)
# team_def = mle.defense_team.transpose(0, 1)
# conf_off = mle.offense_conf.transpose(0, 1)
# conf_def = mle.defense_conf.transpose(0, 1)
#
#
# torch.cat([WMean, WScore, LMean, LScore], 1)
# mle.model.home_field.weight
# mle.model.affine.bias
# mle.model.affine.weight


# MODEL

data_loader = DataLoader(data, batch_size=500, shuffle=True)

model = VAE(data.n_teams, data.n_conf, K)
model.to(DEVICE)
model.train()

# model.initialize(team_off, team_def, conf_off, conf_def)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mse = nn.MSELoss(reduction="sum")

for i in range(100):

    for step in ["E", "M"]:

        for j in range(4):

            for _ in range(10):

                total_loss = 0.0
                total_mse = 0.0

                for WTeamID, LTeamID,  WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ in data_loader:

                    optimizer.zero_grad()

                    for name, param in model.named_parameters():
                        if param.local:
                            param.requires_grad = (step == "E") and param.n == j
                        else:
                            param.requires_grad = (step == "M")

                    score_mean, score_var = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)

                    score = torch.cat([WScore, LScore], 1).to(DEVICE)
                    loss = model.loss(score_mean, score_var, score) / data.n_matches
                    loss.backward()

                    for name, param in model.named_parameters():
                        if step == "M":
                            if param.local:
                                param.grad = None
                        if step == "E":
                            if not param.local:
                                param.grad = None
                            if param.local:
                                if param.n != j:
                                    param.grad = None

                    optimizer.step()

                    total_loss += loss.item()

                    with torch.no_grad():
                        total_mse += mse(score_mean, score).item() * 0.5 / data.n_matches

            print("{:10} {:5} {:5} {:10.6f} {:10.6f}".format(
                i if step=="E" and j==0 else "",
                step if j==0 else "",
                j,
                total_loss,
                total_mse))

print(
    model.encoder_team_off.linear.mean_map.weight.min().item(),
    model.encoder_team_off.linear.mean_map.weight.max().item()
)
print(
    model.encoder_team_off.linear.log_var_map.weight.min().item(),
    model.encoder_team_off.linear.log_var_map.weight.max().item()
)

for name, param in model.named_parameters():
    if not param.local:
        print(name, param.item())


sig2 = torch.exp(model.decoder.gaussian.log_var)
cor = torch.tanh(model.decoder.gaussian.atanh_cor)
Sigma = torch.eye(2) * sig2 * (1. - cor) + cor * sig2
print(Sigma)



torch.cat([score_mean, score], 1)

