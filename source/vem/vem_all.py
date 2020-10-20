import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from source.data.load import MatchDataset
from source.vem.model import VariationalEM

results = dict()
years = range(2004, 2018)
Ks = range(1, 11)
mse = nn.MSELoss(reduction="sum")

for K in Ks:
    results[K] = dict()
    for year in years:
        print(K, year)

        results[K][year] = dict()
        data = MatchDataset(year, "season")
        data_loader = DataLoader(data, batch_size=500, shuffle=False)
        test = MatchDataset(year, "tournament")
        model = VariationalEM(data.n_teams, data.n_conf, K)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        for i in range(10000):

            if i > 2000:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            total_loss = 0.0

            for WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ in data_loader:
                score = torch.cat([WScore, LScore], 1)

                optimizer.zero_grad()

                score_pred = model(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)

                loss = model.loss(score_pred, score) / data.n_matches

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

        # train set
        metrics_train = model.evaluate_average(data, 100)

        # test set
        metrics_test = model.evaluate_average(test, 100)

        params = dict()
        for name, param in model.named_parameters():
            if not param.local:
                params[name] = param.item()

        sig2 = torch.exp(model.decoder.gaussian.log_var)
        cor = torch.tanh(model.decoder.gaussian.atanh_cor)
        params["Sigma"] = (torch.eye(2) * sig2 * (1. - cor) + cor * sig2).detach().numpy()

        params["home_field"] = (model.decoder.home_field.value * model.decoder.gaussian.mult).item()

        results[K][year]["params"] = params
        results[K][year]["metrics_train"] = metrics_train
        results[K][year]["metrics_test"] = metrics_test

        print(results[K][year])


import pandas as pd

Ks_list = list(Ks) * len(years)
years_list = sorted(list(years) * len(Ks))

metrics = pd.DataFrame({"K": Ks_list, "year": years_list})
metrics.set_index(["K", "year"], inplace=True)

columns = [(m, "train") for m in results[1][2010]["metrics_test"].keys()] + \
    [(m, "test") for m in results[1][2010]["metrics_test"].keys()]

for col in columns:
    metrics[col] = 0.

metrics.columns = pd.MultiIndex.from_tuples(columns)

for K in Ks:
    for year in years:
        for k, v in results[K][year]["metrics_train"].items():
            metrics.at[(K, year), (k, "train")] = float(v)
        for k, v in results[K][year]["metrics_test"].items():
            metrics.at[(K, year), (k, "test")] = float(v)


metrics.to_csv("data/results/metrics_vem.csv")

params = pd.DataFrame({"K": Ks_list, "year": years_list})
params.set_index(["K", "year"], inplace=True)

columns = ["home_field", "mean", "mult", "sig2", "cor"]

for col in columns:
    params[col] = 0.


for K in Ks:
    for year in years:
        params.at[(K, year), "home_field"] = results[K][year]["params"]["home_field"]
        params.at[(K, year), "mean"] = results[K][year]["params"]["decoder.gaussian.mean"]
        params.at[(K, year), "mult"] = results[K][year]["params"]["decoder.gaussian.mult"]
        Sigma = results[K][year]["params"]["Sigma"]
        params.at[(K, year), "sig2"] = Sigma[0, 0]
        params.at[(K, year), "cor"] = Sigma[0, 1] / Sigma[0, 0]



params.to_csv("data/results/params_vem.csv")