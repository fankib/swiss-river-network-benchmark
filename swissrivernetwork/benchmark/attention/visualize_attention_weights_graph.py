import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.util import *

# Setup
graph_name = ['swiss-1990', 'swiss-2010'][1]

# Load data
weights = np.load(f"swissrivernetwork/benchmark/dump/attention/weights-{graph_name}.npy") 
g, e = read_graph(graph_name)

print(g, e)
print('shape', e.shape)
#exit()


x = g[:, 0]
y = g[:, 1]

#fix weights in average time:
weights = weights.mean(axis=0)[np.newaxis, :]

# extract graph structure from mean attention weights
A = weights[0]
# Create symmetric Attention:
#A_sym = (A + A.T) / 2        # average both directions
# or
A_sym = np.maximum(A, A.T)   # take the stronger direction

Atop1 = np.argpartition(A_sym, -1, axis=1)[:, -1:]

# create new edges:
e_to = torch.arange(g.shape[0])#.repeat_interleave(2)
e_from = torch.from_numpy(Atop1).flatten()
edge_index = torch.stack([e_from, e_to], dim=0)
print(edge_index)
print('shape:', edge_index.shape)
torch.save((g,e), f'swissrivernetwork/benchmark/dump/graph_{graph_name}-attention.pth') 

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

    # Create symmetric Attention:
    #A_sym = (A + A.T) / 2        # average both directions
    # or
    A_sym = np.maximum(A, A.T)   # take the stronger direction

    Atop1 = np.argpartition(A_sym, -1, axis=1)[:, -1:]


        

    for i in range(A.shape[0]):
        tops = Atop1[i]
        
        # for j in range(A.shape[1]):
        for j in tops:

            # w = A[i, j] # j's contribution to i
            w = A_sym[i, j] 
            #if w <= 0.1:
            #    continue
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