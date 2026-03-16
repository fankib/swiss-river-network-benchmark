import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

graph_name = ['swiss-1990', 'swiss-2010'][0]

# load data
weights = np.load(f"swissrivernetwork/benchmark/dump/attention/weights-{graph_name}.npy")

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
