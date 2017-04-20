import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use("ggplot")

df = pd.read_csv("../dataset.csv")

print df.columns

X = df['points per game'].values
Y = df['assists per game'].values
Z = df['rating'].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=X, ys=Y, zs=Z)

ax.set_xlabel('Points Per Game')
ax.set_ylabel('Assists Per Game')
ax.set_zlabel('Rating')

fig.savefig("../Data Analysis/rating_vs_ppg_assists.png", dpi=200)