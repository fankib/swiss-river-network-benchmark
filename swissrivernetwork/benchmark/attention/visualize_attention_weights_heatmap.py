import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# load data
weights = np.load("swissrivernetwork/benchmark/dump/attention/weights.npy")

t = weights.shape[0]

fig, ax = plt.subplots()

# initial heatmap
hm = sns.heatmap(weights[0], ax=ax, cbar=True)

def update(frame):
    ax.clear()
    sns.heatmap(weights[frame], ax=ax, cbar=False)
    ax.set_title(f"t = {frame}")

ani = FuncAnimation(
    fig,
    update,
    frames=t,
    interval=1000/30  # 30 fps
)

plt.show()
