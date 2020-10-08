import matplotlib.pyplot as plt
from matplotlib import collections as mc
o = model.offense_conf.detach().numpy()
d = model.defense_conf.detach().numpy()

# lines = [[(oo[0], oo[1]), (dd[0], dd[1])] for oo, dd in zip(o, d)]
# lc = mc.LineCollection(lines)
#
# fig, ax = plt.subplots()
# ax.add_collection(lc)
# ax.autoscale()

fig, ax = plt.subplots()
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
for i in range(o.shape[0]):
    ax.arrow(x=d[i, 0], y=d[i, 1], dx=o[i, 0] - d[i, 0], dy=o[i, 1] - d[i, 1], lw=0.5)
    ax.annotate(data.conferences[i], o[i, :], color="red", size=5)

fig.savefig("figs/positions.pdf")
