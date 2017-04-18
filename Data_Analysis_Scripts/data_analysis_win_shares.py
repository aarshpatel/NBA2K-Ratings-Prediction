import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

df = pd.read_csv("../dataset.csv")

print df.columns

ax = df.plot.scatter(x="win shares per 48 minutes", y="rating", figsize=(7,5))
ax.set_title("2k Rating vs Player Win Shares", y=1.08)
ax.set_ylabel("2k Rating")
ax.set_xlabel("Player Win Shares")
fig = ax.get_figure()
plt.tight_layout()
fig.savefig("../Data Analysis/rating_vs_win_shares.png", dpi=200)