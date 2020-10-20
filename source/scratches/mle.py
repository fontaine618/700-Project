import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from source.data.load import MatchDataset
from source.mle.model import ScoreModel

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

results = dict()

for K in [2]:
    results[K] = dict()
    for year in range(2004, 2018):
        print(K, year)
        results[K][year] = dict()
        data = MatchDataset(year, "season")
        data_loader = DataLoader(data, batch_size=500, shuffle=False)
        test = MatchDataset(year, "tournament")
        model = ScoreModel(data.n_teams, data.n_conf, K)
        model.to(DEVICE)
        model.train()
        mse = nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for i in range(1000):
            for step in ["E", "M"]:
                for j in range(1):
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

                        score_pred = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)

                        score = torch.cat([WScore, LScore], 1).to(DEVICE)
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
                        score = torch.cat([WScore, LScore], 1).to(DEVICE)
                        score_pred = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)
                        metrics_train = model.evaluate(score_pred, score)

                    # test set
                    WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ = test[:]
                    with torch.no_grad():
                        score = torch.cat([WScore, LScore], 1).to(DEVICE)
                        score_pred = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)
                        metrics_test = model.evaluate(score_pred, score)

                    # print("{:10} {:2} {:2} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}".format(
                    #     i, step, j,
                    #     metrics_train["llk"], metrics_test["llk"],
                    #     metrics_train["mse"], metrics_test["mse"],
                    #     metrics_train["cross_entropy"], metrics_test["cross_entropy"],
                    #     metrics_train["acc"], metrics_test["acc"]
                    # ))

        params = dict()
        for name, param in model.named_parameters():
            if not param.local:
                params[name] = param.item()


        sig2 = torch.exp(model.gaussian.log_var)
        cor = torch.tanh(model.gaussian.atanh_cor)
        params["Sigma"] = (torch.eye(2) * sig2 * (1. - cor) + cor * sig2).detach().numpy()

        params["home_field"] = (model.home_field.advantage * model.gaussian.mult).item()

        results[K][year]["params"] = params
        results[K][year]["metrics_train"] = metrics_train
        results[K][year]["metrics_test"] = metrics_test




import pandas as pd

K = list([2]) * 14
years = sorted(list(range(2004, 2018)) * 1)

metrics = pd.DataFrame({"K": K, "year": years})
metrics.set_index(["K", "year"], inplace=True)

columns = [(m, "train") for m in results[1][2010]["metrics_test"].keys()] + \
    [(m, "test") for m in results[1][2010]["metrics_test"].keys()]

for col in columns:
    metrics[col] = 0.

metrics.columns = pd.MultiIndex.from_tuples(columns)

for K in [2]:
    for year in range(2004, 2018):
        for k, v in results[K][year]["metrics_train"].items():
            metrics.at[(K, year), (k, "train")] = float(v)
        for k, v in results[K][year]["metrics_test"].items():
            metrics.at[(K, year), (k, "test")] = float(v)


metrics.to_csv("data/results/metrics_mle_2.csv")






K = list([2]) * 14
years = sorted(list(range(2004, 2018)) * 1)


params = pd.DataFrame({"K": K, "year": years})
params.set_index(["K", "year"], inplace=True)

columns = ["home_field", "mean", "mult", "sig2", "cor", "conf_mult"]

for col in columns:
    params[col] = 0.


for K in [2]:
    for year in range(2004, 2018):
        params.at[(K, year), "home_field"] = results[K][year]["params"]["home_field"]
        params.at[(K, year), "mean"] = results[K][year]["params"]["gaussian.mean"]
        params.at[(K, year), "mult"] = results[K][year]["params"]["gaussian.mult"]
        Sigma = results[K][year]["params"]["Sigma"]
        params.at[(K, year), "sig2"] = Sigma[0, 0]
        params.at[(K, year), "cor"] = Sigma[0, 1] / Sigma[0, 0]



params.to_csv("data/results/params_mle_2.csv")