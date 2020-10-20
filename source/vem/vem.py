import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from source.data.load import MatchDataset
from source.vem.model import VariationalEM


data = MatchDataset(2017, "season")
test = MatchDataset(2017, "tournament")
data_loader = DataLoader(data, batch_size=500, shuffle=True)

K = 2

# INITIALIZATION

from source.models.latent_variable_model import OffDefTeamConfLVM
from source.models.score_model import ScoreModel
from source.models.factors import Distance

space_model = Distance()

mle = OffDefTeamConfLVM(data.n_teams, data.n_conf, K, ScoreModel(space_model))
mle.train()
mse = nn.MSELoss(reduction="sum")

optimizer = torch.optim.SGD(mle.parameters(), lr=0.01, momentum=0.5)


for i in range(200):

    total_loss = 0.0

    for WTeamID, LTeamID,  WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ in data_loader:

        optimizer.zero_grad()

        WMean, LMean = mle.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)
        loss = (mse(WMean, WScore) + mse(LMean, LScore)) * 0.5 / data.WLoc.size(0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            mle.project()

    print("{:10} {:10.3f}".format(i, total_loss))


team_off = mle.offense_team.transpose(0, 1)
team_def = mle.defense_team.transpose(0, 1)
conf_off = mle.offense_conf.transpose(0, 1)
conf_def = mle.defense_conf.transpose(0, 1)


# MODEL


model = VariationalEM(data.n_teams, data.n_conf, K)

model.initialize(team_off, team_def, conf_off, conf_def)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

mse = nn.MSELoss(reduction="sum")

exp_mse = 0.0

for i in range(10000):

    for WTeamID, LTeamID,  WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ in data_loader:
        score = torch.cat([WScore, LScore], 1)

        optimizer.zero_grad()

        score_pred = model(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)

        loss = model.loss(score_pred, score) / data.n_matches

        loss.backward()

        optimizer.step()

    if i % 100 == 0:
        # train set
        metrics_train = model.evaluate_average(data, 100)

        # test set
        metrics_test = model.evaluate_average(test, 100)

        print("{:10} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}".format(
            i,
            metrics_train["llk"], metrics_test["llk"],
            metrics_train["mse"], metrics_test["mse"],
            metrics_train["cross_entropy"], metrics_test["cross_entropy"],
            metrics_train["acc"], metrics_test["acc"]
        ))


torch.save(model, "./data/results/VariationalEM_2")
# train set
metrics_train = model.evaluate_average(data, 100)

# test set
metrics_test = model.evaluate_average(test, 100)

print("{:10} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}".format(
    i,
    metrics_train["llk"], metrics_test["llk"],
    metrics_train["mse"], metrics_test["mse"],
    metrics_train["cross_entropy"], metrics_test["cross_entropy"],
    metrics_train["acc"], metrics_test["acc"]
))






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



torch.cat([score_pred, score], 1)

