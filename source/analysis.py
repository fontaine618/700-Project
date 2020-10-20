import pandas as pd

metrics = pd.read_csv("data/results/metrics_vem.csv", header=[0, 1], index_col=[0, 1])

metrics.reset_index(inplace=True)

metrics = metrics.loc[metrics[("mse", "train")] != 0.]
metrics_summary = metrics.groupby("K").agg(["min", "max", "median"])

params = pd.read_csv("data/results/params_mle_2.csv", header=[0], index_col=[0, 1])
params.reset_index(inplace=True)

pd.options.display.max_rows=1000
pd.options.display.max_columns=1000

# Metrics plot
import matplotlib.pyplot as plt
plt.style.use("seaborn")

color_train = "#00274C"
color_test = "#FFCB05"

fig, axs = plt.subplots(2, 2)

for i, metric in enumerate(["llk", "mse", "acc", "cross_entropy"]):
    main = {"llk": "Log Likelihood", "mse": "Mean Squared Error",
            "acc": "Prediction Accuracy", "cross_entropy": "Binary Cross-entropy"}[metric]

    row = int(i / 2)
    col = i - row * 2

    print(row, col)

    ax = axs[row, col]

    # train
    ax.fill_between(x=range(1, 11),
                    y1=metrics_summary[(metric, "train", "min")],
                    y2=metrics_summary[(metric, "train", "max")],
                    color=color_train, alpha=0.2)
    ax.plot(range(1, 11), metrics_summary[(metric, "train", "median")], color=color_train, label="Train")

    # test
    ax.fill_between(x=range(1, 11),
                    y1=metrics_summary[(metric, "test", "min")],
                    y2=metrics_summary[(metric, "test", "max")],
                    color=color_test, alpha=0.2)
    ax.plot(range(1, 11), metrics_summary[(metric, "test", "median")], color=color_test, label="Test")

    ax.set_ylabel(main)

axs[0, 0].set_xticklabels("")
axs[0, 1].set_xticklabels("")
axs[1, 0].set_xlabel("Nb. Components")
axs[1, 1].set_xlabel("Nb. Components")

axs[1, 1].legend(loc="upper left")

plt.tight_layout()

fig.savefig("figs/metrics_vem.pdf")



# Metrics plot
import matplotlib.pyplot as plt
plt.style.use("seaborn")

color_train = "#00274C"
color_test = "#FFCB05"

fig, axs = plt.subplots(4, 1, figsize=(4, 7))

for i, metric in enumerate(["llk", "mse", "acc", "cross_entropy"]):
    main = {"llk": "Log Likelihood", "mse": "Mean Squared Error",
            "acc": "Prediction Accuracy", "cross_entropy": "Binary Cross-entropy"}[metric]

    ax = axs[i]

    # train
    ax.fill_between(x=range(1, 11),
                    y1=metrics_summary[(metric, "train", "min")],
                    y2=metrics_summary[(metric, "train", "max")],
                    color=color_train, alpha=0.2)
    ax.plot(range(1, 11), metrics_summary[(metric, "train", "median")], color=color_train, label="Train")

    # test
    ax.fill_between(x=range(1, 11),
                    y1=metrics_summary[(metric, "test", "min")],
                    y2=metrics_summary[(metric, "test", "max")],
                    color=color_test, alpha=0.2)
    ax.plot(range(1, 11), metrics_summary[(metric, "test", "median")], color=color_test, label="Test")

    ax.set_ylabel(main)

for i in range(3):
    axs[i].set_xticklabels("")
axs[3].set_xlabel("Nb. Components")

axs[0].legend(loc="upper right")

plt.tight_layout()

fig.savefig("figs/metrics_vem.pdf")




# params plot
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use("seaborn")

blue = "#00274C"
maize = "#FFCB05"

params = params[params["K"] == 2]


fig, axs = plt.subplots(2, 2)
minorLocator = MultipleLocator(2)
# home field
ax = axs[0, 0]
ax.plot(params["year"].astype(int), 2. * params["home_field"], color=blue)
ax.set_ylabel("Home-field adv.")
ax.set_ylim([0., 5.])
ax.xaxis.set_major_locator(minorLocator)
# mean
ax = axs[0, 1]
ax.plot(params["year"].astype(int), params["mean"], color=blue)
ax.set_ylabel("Mean score")
ax.set_ylim([0., 80.])
ax.xaxis.set_major_locator(minorLocator)
# home field
ax = axs[1, 0]
ax.plot(params["year"].astype(int), params["sig2"], color=blue)
ax.set_ylabel("Variance")
ax.set_ylim([0., 150.])
ax.xaxis.set_major_locator(minorLocator)
# home field
ax = axs[1, 1]
ax.plot(params["year"].astype(int), params["cor"], color=blue)
ax.set_ylabel("Correlation")
ax.set_ylim([0., 1.])
ax.xaxis.set_major_locator(minorLocator)


plt.tight_layout()
fig.savefig("figs/params_mle.pdf")





# params plot
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use("seaborn")

blue = "#00274C"
maize = "#FFCB05"

params = params[params["K"] == 2]


fig, axs = plt.subplots(4, 1, figsize=(4, 7))
minorLocator = MultipleLocator(2)
# home field
ax = axs[0]
ax.plot(params["year"].astype(int), 2. * params["home_field"], color=blue)
ax.set_ylabel("Home-field adv.")
ax.set_ylim([0., 5.])
ax.xaxis.set_major_locator(minorLocator)
# mean
ax = axs[1]
ax.plot(params["year"].astype(int), params["mean"], color=blue)
ax.set_ylabel("Mean score")
ax.set_ylim([0., 80.])
ax.xaxis.set_major_locator(minorLocator)
# home field
ax = axs[2]
ax.plot(params["year"].astype(int), params["sig2"], color=blue)
ax.set_ylabel("Variance")
ax.set_ylim([0., 150.])
ax.xaxis.set_major_locator(minorLocator)
# home field
ax = axs[3]
ax.plot(params["year"].astype(int), params["cor"], color=blue)
ax.set_ylabel("Correlation")
ax.set_ylim([0., 1.])
ax.xaxis.set_major_locator(minorLocator)

for i in range(3):
    axs[i].set_xticklabels("")

plt.tight_layout()
fig.savefig("figs/params_mle.pdf")