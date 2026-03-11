import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.util import *

# Setup
graph_name = "swiss-1990"

# Load data
weights = np.load("swissrivernetwork/benchmark/dump/attention/weights.npy")  # (t, 28, 28)
g, e = read_graph(graph_name)

x = g[:, 0]
y = g[:, 1]

t = weights.shape[0]

# colormap setup
norm = colors.Normalize(vmin=weights.min(), vmax=weights.max())
cmap = cm.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6, 6))

# draw nodes once
ax.scatter(x, y, s=60, color="black", zorder=3)
ax.set_aspect("equal")
ax.axis("off")

# draw node labels
for i, label in enumerate(g[:, 2]):
    ax.text(x[i], y[i], str(label.item()), fontsize=10, ha="left", va="top", zorder=4)

# prepare a list to hold arrow artists
arrows = []

def update(frame):
    global arrows
    # remove previous arrows
    for arr in arrows:
        arr.remove()
    arrows = []

    A = weights[frame]

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            w = A[i, j] # j's contribution to i
            if w <= 0.1:
                continue
            color = cmap(norm(w))
            lw = 5*norm(w)
            a = max(0, -0.1 + 1.1*norm(w))

            arr = ax.annotate(
                "",
                xytext=(x[j], y[j]),
                xy=(x[i], y[i]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=lw,
                    alpha=a
                ),
                zorder=2
            )
            arrows.append(arr)

    ax.set_title(f"t = {frame}")

ani = FuncAnimation(fig, update, frames=t, interval=10)

# optional: save
# ani.save("attention_graph.mp4", fps=30)

plt.show()