import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

df = pd.read_csv("../dataset.csv")

print df.columns

ax = df.plot.scatter(x="usage percentage", y="rating", figsize=(7,5))
ax.set_title("2k Rating vs Player Usage Percentage", y=1.08)
ax.set_ylabel("2k Rating")
ax.set_xlabel("Player Usage Percentage")
fig = ax.get_figure()
plt.tight_layout()
fig.savefig("../Data Analysis/rating_vs_usage.png", dpi=200)