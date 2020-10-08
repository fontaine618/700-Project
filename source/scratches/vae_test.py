import torch
from source.vae.factors import Gaussian, L2Norm, Subtract, Prior, LinearMeanVariance
from source.vae.decoder import MeanModel, ScoreModel
from source.vae.encoder import Encoder


x = torch.eye(8)

encoder = Encoder(8, 3)

encoder(x)

decoder = ScoreModel()

n_team = 10
win_team_id = torch.randint(0, 10, (5, 1))



mean0 = torch.randn(4, 1)
mean1 = torch.randn(4, 1)
var0 = torch.rand(4, 1)
var1 = torch.rand(4, 1)

self = Gaussian()

self = L2Norm()

mean = torch.randn((4, 2))
var = torch.rand((4, 2))

dim = 1

mean0 = torch.randn(4, 2)
mean1 = torch.randn(4, 2)
mean2 = torch.randn(4, 2)
mean3 = torch.randn(4, 2)
var0 = torch.rand(4, 2)
var1 = torch.rand(4, 2)
var2 = torch.rand(4, 2)
var3 = torch.rand(4, 2)

self = MeanModel()

m0, v0 = self(mean0, var0, mean1, var1, mean2, var2, mean3, var3)
m1, v1 = self(mean0, var0, mean1, var1, mean2, var2, mean3, var3)




mean0 = torch.randn(4, 1, requires_grad=True)
mean1 = torch.randn(4, 1, requires_grad=True)
var0 = torch.rand(4, 1, requires_grad=True)
var1 = torch.rand(4, 1, requires_grad=True)
self = Gaussian()

mean, var = self(mean0, var0, mean1, var1)

loss = torch.sum(mean)
loss.backward()
self.var.grad



win_team_off_mean, win_team_off_var = torch.randn((4, 2)), torch.rand((4, 2))
win_conf_off_mean, win_conf_off_var = torch.randn((4, 2)), torch.rand((4, 2))
los_team_def_mean, los_team_def_var = torch.randn((4, 2)), torch.rand((4, 2))
los_conf_def_mean, los_conf_def_var = torch.randn((4, 2)), torch.rand((4, 2))
los_team_off_mean, los_team_off_var = torch.randn((4, 2)), torch.rand((4, 2))
los_conf_off_mean, los_conf_off_var = torch.randn((4, 2)), torch.rand((4, 2))
win_team_def_mean, win_team_def_var = torch.randn((4, 2)), torch.rand((4, 2))
win_conf_def_mean, win_conf_def_var = torch.randn((4, 2)), torch.rand((4, 2))
win_loc = torch.randint(low=-1, high=2, size=(4, 1))
los_loc = - win_loc


self = ScoreModel()

score_mean, score_var, diff_mean, diff_var = self.forward(
    win_team_off_mean, win_team_off_var,
    win_conf_off_mean, win_conf_off_var,
    los_team_def_mean, los_team_def_var,
    los_conf_def_mean, los_conf_def_var,
    los_team_off_mean, los_team_off_var,
    los_conf_off_mean, los_conf_off_var,
    win_team_def_mean, win_team_def_var,
    win_conf_def_mean, win_conf_def_var,
    win_loc, los_loc
)


mean0 = torch.randn(4, 1)
mean1 = torch.randn(4, 1)


x = torch.eye(8)
self = LinearMeanVariance(8, 2)
mean, var = self(x)

self = Prior()
self(mean, var)