import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from source.data.load import MatchDataset
from source.models.latent_variable_model import OffDefTeamConfLVM
from source.models.score_model import ScoreModel
from source.models.factors import InnerProduct, Distance

season_results = MatchDataset(2017, "season")
tournament_results = MatchDataset(2017, "tournament")

data = MatchDataset(2017, "season")

data_loader = DataLoader(data, batch_size=500, shuffle=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

space_model = Distance()
# space_model = InnerProduct()

model = OffDefTeamConfLVM(data.n_teams, data.n_conf, 10, ScoreModel(space_model)).to(DEVICE)
model.train()
mse = nn.MSELoss(reduction="sum")

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


for i in range(100):

    total_loss = 0.0

    for WTeamID, LTeamID,  WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore, _, _ in data_loader:
        WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc, WScore, LScore = \
            WTeamID.to(DEVICE), LTeamID.to(DEVICE), WTeamConf.to(DEVICE), LTeamConf.to(DEVICE), \
            WLoc.to(DEVICE), LLoc.to(DEVICE), WScore.to(DEVICE), LScore.to(DEVICE)

        optimizer.zero_grad()

        WMean, LMean = model.forward(WTeamID, LTeamID, WTeamConf, LTeamConf, WLoc, LLoc)
        loss = (mse(WMean, WScore) + mse(LMean, LScore)) * 0.5 / data.WLoc.size(0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            model.project()

    print("{:10} {:10.3f}".format(i, total_loss))


torch.cat([WMean, WScore, LMean, LScore], 1)
model.model.home_field.weight
model.model.affine.bias
model.model.affine.weight

