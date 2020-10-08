import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from source.data.load import MatchDataset
from source.models.factors import SumAlong, InnerProduct

season_results = MatchDataset(2017, "season")
tournament_results = MatchDataset(2017, "tournament")

self = MatchDataset(2017, "season")

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
DEVICE = get_device()

_, WTeamID, LTeamID, WScore, LScore, WLoc, LLoc, WTeamConf, LTeamConf, _, _ = season_results[:]

# dimensions
K = 4

# initialize latent variables
torch.manual_seed(0)

Z_team_off = torch.randn((season_results.n_teams, K), requires_grad=True)
Z_team_def = torch.randn((season_results.n_teams, K), requires_grad=True)

Z_conf_off = torch.randn((season_results.n_conf, K), requires_grad=True)
Z_conf_def = torch.randn((season_results.n_conf, K), requires_grad=True)

# initialize home-field adv
hf = torch.rand(1, requires_grad=True)

# initialize linear map
intercept = torch.tensor(72. / 3., requires_grad=True)
slope = torch.tensor(10. / 3., requires_grad=True)


def nearest_orthogonal(Z):
    u, _, v = torch.svd(Z)
    return torch.mm(u, v.t())

STEP_SIZE = 1.0e-7

prev_error = 1e10

sum_along = SumAlong()
inner_product = InnerProduct()
home_field = torch.nn.Linear(1, 1, bias=False)
affine = torch.nn.Linear(1, 1)

WLoc = torch.reshape(WLoc, (-1, 1)).float()
LLoc = torch.reshape(LLoc, (-1, 1)).float()

for epoch in range(1, 1000):
    ZW_off = sum_along(Z_team_off, Z_conf_off, WTeamID, WTeamConf)
    ZW_def = sum_along(Z_team_def, Z_conf_def, WTeamID, WTeamConf)
    ZL_off = sum_along(Z_team_off, Z_conf_off, LTeamID, LTeamConf)
    ZL_def = sum_along(Z_team_def, Z_conf_def, LTeamID, LTeamConf)
    W = inner_product(ZW_off, ZL_def) + home_field(WLoc)
    L = inner_product(ZL_off, ZW_def) + home_field(LLoc)
    WMean = affine(W)
    LMean = affine(L)


    # compute mean
    Z_W_prod = torch.sum(torch.exp((Z_team_off[WTeamID, :] + Z_conf_off[WTeamConf, :])) * \
                torch.exp((Z_team_def[LTeamID, :] + Z_conf_def[LTeamConf, :])), 1)
    W_mean = Z_W_prod + hf * WLoc
    W_mean = intercept + slope * W_mean

    Z_L_prod = torch.sum((torch.exp(Z_team_off[LTeamID, :] + Z_conf_off[LTeamConf, :])) * \
                torch.exp((Z_team_def[WTeamID, :] + Z_conf_def[WTeamConf, :])), 1)
    L_mean = Z_L_prod + hf * LLoc
    L_mean = intercept + slope * L_mean

    # error
    W_error = F.mse_loss(W_mean, WScore.float())
    L_error = F.mse_loss(L_mean, LScore.float())
    error = W_error + L_error
    error.backward()

    print(epoch, torch.sqrt(error).item(), STEP_SIZE)
    # GD step
    with torch.no_grad():
        hf -= STEP_SIZE * hf.grad
        intercept -= STEP_SIZE * intercept.grad
        slope -= STEP_SIZE * slope.grad
        Z_team_off -= STEP_SIZE * Z_team_off.grad
        Z_team_def -= STEP_SIZE * Z_team_def.grad
        Z_conf_off -= STEP_SIZE * Z_conf_off.grad
        Z_conf_def -= STEP_SIZE * Z_conf_def.grad
        Z_team_off = nearest_orthogonal(Z_team_off)
        Z_team_def = nearest_orthogonal(Z_team_def)
        Z_conf_off = nearest_orthogonal(Z_conf_off)
        Z_conf_def = nearest_orthogonal(Z_conf_def)
    hf.requires_grad = True
    intercept.requires_grad = True
    slope.requires_grad = True
    Z_team_off.requires_grad = True
    Z_team_def.requires_grad = True
    Z_conf_off.requires_grad = True
    Z_conf_def.requires_grad = True

    # if torch.sqrt(error).item() > prev_error:
    #     STEP_SIZE *= 0.9
    prev_error = torch.sqrt(error).item()


pd.DataFrame({
    "conf": season_results.conferences,
    "off": torch.sum(Z_conf_off, 1).detach().numpy(),
    "def":  torch.sum(Z_conf_def, 1).detach().numpy()
}).sort_values(by="off", ascending=False)



