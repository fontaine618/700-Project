import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from source.data.load import MatchDataset
from source.mle.model import ScoreModel

years = range(2004, 2018)
K = 2

for year in years:
    data = MatchDataset(year, "season")
    data_loader = DataLoader(data, batch_size=500, shuffle=False)
    test = MatchDataset(year, "tournament")
    model = ScoreModel(data.n_teams, data.n_conf, K)
    model.train()
    mse = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for i in range(1000):
        total_loss = 0.0
        total_mse = 0.0

        for WTeamID, LTeamID,  WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ in data_loader:

            optimizer.zero_grad()

            score_pred = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)

            score = torch.cat([WScore, LScore], 1)
            loss = model.loss(score_pred, score) / data.n_matches
            loss.backward()

            optimizer.step()

            model.project()

            total_loss += loss.item()

            with torch.no_grad():
                total_mse += mse(score_pred, score).item() * 0.5 / data.n_matches


        # train set
        WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ = data[:]
        with torch.no_grad():
            score = torch.cat([WScore, LScore], 1)
            score_pred = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)
            metrics_train = model.evaluate(score_pred, score)

        # test set
        WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ = test[:]
        with torch.no_grad():
            score = torch.cat([WScore, LScore], 1)
            score_pred = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)
            metrics_test = model.evaluate(score_pred, score)



        print("{:10} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}".format(
            i,
            metrics_train["llk"], metrics_test["llk"],
            metrics_train["mse"], metrics_test["mse"],
            metrics_train["cross_entropy"], metrics_test["cross_entropy"],
            metrics_train["acc"], metrics_test["acc"]
        ))


    torch.save(model, "./data/results/MLE_2/"+str(year))


year = 2014
model = torch.load("./data/results/MLE_2/"+str(year))
# latent variables
data = MatchDataset(year, "season")
data_loader = DataLoader(data, batch_size=500, shuffle=False)
test = MatchDataset(year, "tournament")

conf_id = torch.tensor(data.team_conf_dict["ConfID"].values).long()



# predictions

print(torch.cat([score_pred, score], 1))


# parameters

for name, param in model.named_parameters():
    if not param.local:
        print(name, param)

sig2 = torch.exp(model.gaussian.log_var)
cor = torch.tanh(model.gaussian.atanh_cor)
print((torch.eye(2) * sig2 * (1. - cor) + cor * sig2).detach().numpy())

# get all matches

matches = [[t0, t1] for t1 in range(data.n_teams) for t0 in range(data.n_teams) if t0 > t1]
ID = torch.tensor(matches)

ID0 = ID[:, 0]
ID1 = ID[:, 1]

ConfID0 = torch.tensor(data.team_conf_dict.loc[data.teams[ID0.numpy()], "ConfID"].values).view(-1, 1).long()
ConfID1 = torch.tensor(data.team_conf_dict.loc[data.teams[ID1.numpy()], "ConfID"].values).view(-1, 1).long()

ID0 = ID0.view(-1, 1).long()
ID1 = ID1.view(-1, 1).long()

Loc0 = torch.zeros_like(ID0).float()
Loc1 = torch.zeros_like(ID0).float()

score_pred = model.forward(ID0, ID1, ConfID0, ConfID1, Loc0, Loc1)

winner = (score_pred[:, 0] > score_pred[:, 1]).float().view(-1, 1)
winner = torch.cat([winner, 1. - winner], 1)

wins = torch.zeros((data.n_teams, 1))
pf = torch.zeros((data.n_teams, 1))
pa = torch.zeros((data.n_teams, 1))

for m in range(score_pred.size(0)):
    wins[ID0[m]] += winner[m, 0]
    wins[ID1[m]] += winner[m, 1]
    pf[ID0[m]] += score_pred[m, 0]
    pf[ID1[m]] += score_pred[m, 1]
    pa[ID0[m]] += score_pred[m, 1]
    pa[ID1[m]] += score_pred[m, 0]

win_pct = wins.view(-1).detach().numpy()
pf = pf.view(-1).detach().numpy()
pa = pa.view(-1).detach().numpy()

import pandas as pd
win_pct_df = pd.DataFrame(data={
    "name":data.team_names,
    "win": win_pct,
    "pf": pf/350.,
    "pa": pa/350.},
    index=data.teams)

print(win_pct_df.sort_values("win", ascending=False).iloc[0:30].to_latex())
print(win_pct_df.sort_values("win", ascending=False).iloc[-5:].to_latex())





# conference matches


matches = [[c0, c1] for c1 in range(data.n_conf) for c0 in range(data.n_conf) if c0 > c1]
ID = torch.tensor(matches)

ConfID0 = ID[:, 0].view(-1, 1).long()
ConfID1 = ID[:, 1].view(-1, 1).long()

ID0 = torch.zeros_like(ConfID0).long()
ID1 = torch.zeros_like(ConfID1).long()

Loc0 = torch.zeros_like(ID0).float()
Loc1 = torch.zeros_like(ID0).float()

score_pred = model.forward(ID0, ID1, ConfID0, ConfID1, Loc0, Loc1)

winner = (score_pred[:, 0] > score_pred[:, 1]).float().view(-1, 1)
winner = torch.cat([winner, 1. - winner], 1)

wins = torch.zeros((data.n_conf, 1))

for m in range(score_pred.size(0)):
    wins[ConfID0[m]] += winner[m, 0]
    wins[ConfID1[m]] += winner[m, 1]

win_pct = wins.view(-1).detach().numpy()

import pandas as pd

win_pct_df = pd.DataFrame(data={
    "abbrev": data.conferences,
    "name": data.conference_names.loc[data.conferences.values].values.reshape(-1),
    "win": win_pct.astype(int)})

win_pct_df["name"] = win_pct_df["name"].str.replace(" Conference", "")

print(win_pct_df.sort_values("win", ascending=False)[["name", "win"]].to_latex(index=False))














# plot conf latent positions

import matplotlib.pyplot as plt
plt.style.use("seaborn")

o = model.conf_off.detach().numpy()
d = model.conf_def.detach().numpy()

fig, ax = plt.subplots(1, 1, figsize=(4, 3))


ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)

i0 = 8
i1 = 6


for i in range(o.shape[0]):
    ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
             lw=1., alpha=0.1, head_width=0.01)

for i in [i0, i1]:
    ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
             lw=1., alpha=1.0, head_width=0.01,
             color="#00274C" if i==i0 else "#FFCB05")


ax.annotate(data.conferences[i0], [-0.1, 0.3], color="#00274C", size=14)
ax.annotate(data.conferences[i1], [0.2, 0.0], color="#FFCB05", size=14)

ax.arrow(x=0.2, y=-0.4, dx=0.2, dy=0., lw=1., alpha=1.0, head_width=0.01)
ax.annotate("off", [0.4, -0.38], color="black", size=10)
ax.annotate("def", [0.2, -0.38], color="black", size=10)

ax.scatter([0], [0], color="black")

#ax.set_title("Conference latent positions")
plt.tight_layout()

fig.savefig("figs/positions_mle_2_conf.pdf")



# plot team + conf
ConfID = torch.tensor(
    data.team_conf_dict.loc[data.teams[range(data.n_teams)], "ConfID"].values
).long()

o, d = model.predict_team_plus_conference(ConfID)
o = o.detach().numpy()
d = d.detach().numpy()

fig, ax = plt.subplots(figsize=(4, 3))

ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.7, 0.7)


for i in range(o.shape[0]):
    ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
             lw=1., alpha=0.1, head_width=0.01)

for i in [168, 100]:
    ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
             lw=1., alpha=1.0, head_width=0.01)
    team_id = data.teams[i]
    ax.annotate(data.team_names[team_id], o[i, :], color="black", size=10)

ax.arrow(x=0.2, y=-0.4, dx=0.2, dy=0., lw=1., alpha=1.0, head_width=0.01)
ax.annotate("off", [0.4, -0.38], color="black", size=10)
ax.annotate("def", [0.2, -0.38], color="black", size=10)

ax.scatter([0], [0], color="black")

ax.set_title("Team + Conference latent positions")

plt.tight_layout()
fig.savefig("figs/positions_mle_2_team_conf.pdf")





# best / worst teams
ConfID = torch.tensor(
    data.team_conf_dict.loc[data.teams[range(data.n_teams)], "ConfID"].values
).long()

o, d = model.predict_team_plus_conference(ConfID)
o = o.detach().numpy()
d = d.detach().numpy()


best = win_pct_df.sort_values("win", ascending=False).iloc[0:25]
worst = win_pct_df.sort_values("win", ascending=True).iloc[0:25]



fig, ax = plt.subplots(figsize=(4, 3))

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)


for i in range(o.shape[0]):
    ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
             lw=1., alpha=0.1, head_width=0.01)
for i in range(o.shape[0]):
    team_id = data.teams[i]
    if team_id in best.index:
        ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
                 lw=1., alpha=1., head_width=0.01, color="#00274C")
    if team_id in worst.index:
        ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
                 lw=1., alpha=1., head_width=0.01, color="#FFCB05")


ax.annotate("Top 25", [0.0, 0.4], color="#00274C", size=14)
ax.annotate("Bottom 25", [0.2, -0.3], color="#FFCB05", size=14)

ax.scatter([0], [0], color="black")

plt.tight_layout()
fig.savefig("figs/positions_mle_2_team_conf_best_worst.pdf")





# by conferences
ConfID = torch.tensor(
    data.team_conf_dict.loc[data.teams[range(data.n_teams)], "ConfID"].values
).long()

o, d = model.predict_team_plus_conference(ConfID)
o = o.detach().numpy()
d = d.detach().numpy()


fig, ax = plt.subplots(figsize=(4, 3))

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)


for i in range(o.shape[0]):
    ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
             lw=1., alpha=0.1, head_width=0.01)
for i in range(o.shape[0]):
    conf_id = ConfID[i]
    conf_name = data.conferences[conf_id]
    if conf_id == 8:
        ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
                 lw=1., alpha=1., head_width=0.01, color="#00274C")
    if conf_id == 6:
        ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1],
                 lw=1., alpha=1., head_width=0.01, color="#FFCB05")


ax.annotate(data.conferences[8], [-0.1, 0.3], color="#00274C", size=14)
ax.annotate(data.conferences[6], [0.25, 0.0], color="#FFCB05", size=14)

ax.scatter([0], [0], color="black")

ax.arrow(x=0.2, y=-0.4, dx=0.2, dy=0., lw=1., alpha=1.0, head_width=0.01)
ax.annotate("off", [0.4, -0.38], color="black", size=10)
ax.annotate("def", [0.2, -0.38], color="black", size=10)

plt.tight_layout()
fig.savefig("figs/positions_mle_2_team_conf_by_conf.pdf")