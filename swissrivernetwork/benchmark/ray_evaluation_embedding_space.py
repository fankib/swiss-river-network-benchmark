import os
import math

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from ray.tune import ExperimentAnalysis

from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.lstm_embedding import EMBEDDING_MODEL_FACTORY
from swissrivernetwork.benchmark.train_isolated_station import read_stations, read_graph
from swissrivernetwork.benchmark.test_embedding_model import test_lstm_embedding

from swissrivernetwork.gbr25.graph_exporter import plot_graph

from swissrivernetwork.benchmark.ray_evaluation_embedding import experiment_analysis_single_model, experiment_analysis_lowd, experiment_analysis_static_embedding

VERBOSE = True

def parameter_count(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print('TOTAL MODEL PARAMETERS: ', pytorch_total_params)
    return pytorch_total_params

def normalize_to_target(coords, target):
    """
    Normalize coords to the same min/max range as target (per dimension).
    """
    target_min = target.min(axis=0)
    target_max = target.max(axis=0)
    
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    
    # Avoid division by zero
    scale = np.where(coords_max - coords_min == 0, 1, coords_max - coords_min)
    
    normalized = (coords - coords_min) / scale  # scale to 0-1
    normalized = normalized * (target_max - target_min) + target_min  # scale to target range
    return normalized

def evaluate_best_trial_isolated_station(graph_name, method, station, i):    

    #analysis = experiment_analysis_single_model(graph_name, method)
    #analysis = experiment_analysis_lowd(graph_name, method)
    analysis = experiment_analysis_static_embedding(graph_name, method)

    
    #df = analysis.dataframe()
    # Get the best Trial:
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config
    #VERBOSE and print(f"Best trial: {best_trial}")
    VERBOSE and print(f'Best Trial Configuration: {best_config}')
    
    # Get the best checkpoint
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")    
    
    # Create Model:     
    num_embeddings = len(read_stations(graph_name))
    model_factory = EMBEDDING_MODEL_FACTORY[method]
    model = model_factory(1, num_embeddings, best_config['embedding_size'], best_config['hidden_size'])

    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))    
    model.eval()

    # Examinate Embedding Space -- hackedihackhack
    embeddings = model.embedding.weight.detach().cpu().numpy()

    # Gaussian embeddings:
    # stats of embeddings:
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    print('mean', mean)
    print('std', std)    
    noise = 0.1
    embeddings_noise = embeddings + np.random.normal(0, noise*std, size=embeddings.shape)

    if embeddings.shape[1] == 2:
        reduced = embeddings
    #elif embeddings.shape[1] == 3:
        #reduced = embeddings[:, 0:2]
    #    reduced = embeddings[:, [0, 2]]
    else:
        USE_PCA_XOR_tSNE = False
        if USE_PCA_XOR_tSNE:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)
        else: # use t-SNE
            #from sklearn.manifold import TSNE            
            #tsne = TSNE(n_components=2, random_state=42, perplexity=4, metric="cosine")
            #reduced = tsne.fit_transform(embeddings)

            from umap import UMAP
            #umap = UMAP(n_components=2, random_state=42, n_neighbors=4, metric="cosine")                        
            umap = UMAP(n_components=2, random_state=42, n_neighbors=5)                        
            reduced = umap.fit_transform(embeddings)
    
    # plot the space:
    #plt.figure(figsize=(8, 6))    

    # Plot Edges
    x,e = read_graph(graph_name)
    coords = x.numpy()[:, 0:2]
    coords = normalize_to_target(coords, reduced) # map coordinates into embedding space
    stations = read_stations(graph_name)
    #colors = node_colors(graph_name, method)
    #reduced = x # plot coordinates! (Debug)

    frames = 100  # number of animation frames

    # Create figure
    df = pd.read_csv(f'swissrivernetwork/benchmark/dump/test_results/{graph_name}_{method}.csv')    
    data = df['RMSE'].values    
    norm = plt.Normalize(vmin=0.5, vmax=1.2)
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:,0], coords[:,1], alpha=0.7, zorder=2, s=100, c=data, cmap=cmap, norm=norm)

    # Prepare edge lines
    lines = []
    for start, end in e.T:
        line, = ax.plot([], [], 'k-', alpha=0.8, linewidth=0.8, zorder=1)
        lines.append(line)

    # Prepare text labels
    texts = []
    for i, text in enumerate(stations):
        txt = ax.text(coords[i, 0], coords[i, 1], text, fontsize=9)
        texts.append(txt)

    # Set axis limits
    x_min = min(coords[:,0].min(), reduced[:,0].min())
    x_max = max(coords[:,0].max(), reduced[:,0].max())
    y_min = min(coords[:,1].min(), reduced[:,1].min())
    y_max = max(coords[:,1].max(), reduced[:,1].max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Linear interpolation
    def interpolate(start, end, alpha):
        return start + alpha * (end - start)

    # Animation update
    def update(frame):
        alpha = min(1, max(0, (1.4*frame/(frames-1))-0.2))
        positions = interpolate(coords, reduced, alpha)
        
        # Update nodes
        sc.set_offsets(positions)
        
        # Update edges
        for line, (start, end) in zip(lines, e.T):
            x_coords = [positions[start, 0], positions[end, 0]]
            y_coords = [positions[start, 1], positions[end, 1]]
            line.set_data(x_coords, y_coords)
        
        # Update text positions
        for i, txt in enumerate(texts):
            txt.set_position((positions[i, 0], positions[i, 1]))
        
        return sc, *lines, *texts

    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=10, blit=True, repeat=False)
    anim.save(f'swissrivernetwork/benchmark/dump/embedding_space/embedding_{graph_name}_{method}.gif', writer='pillow')  

    writer = FFMpegWriter(fps=30, bitrate=1800)
    anim.save(f'swissrivernetwork/benchmark/dump/embedding_space/embedding_{graph_name}_{method}.mp4', writer=writer)

    plt.title(f"Animate {graph_name} for {method}")
    plt.show()
    fig.savefig(f'swissrivernetwork/benchmark/dump/embedding_space/embedding_{graph_name}_{method}.png', dpi=300)
    # To save: anim.save('graph_animation.gif', writer='pillow')  

    # model summary
    total_params = parameter_count(model)

         
    return total_params


def process_method(graph_name, method):
    
    print(f'~~~ Process {method} on {graph_name} ~~~')

    failed_stations= []

    # Setup
    stations = read_stations(graph_name)
    # statistics:
    print('Expected Stations: ', len(stations))

    for i,station in enumerate(stations):
        if station in failed_stations:
            continue # fix this stations!
        
        #if True:
        try:            
            total_params = evaluate_best_trial_isolated_station(graph_name, method, station, i)          
            break

        except Exception as e:            
            print(f'[ERROR] Station {station} failed! Reason: {e}')
            failed_stations.append(station)
    
    print('METHOD LEARNABLE PARAMETERS:', total_params)

    print('FAILED_STATIONS:', failed_stations)


    


if __name__ == '__main__':

    GRAPH_NAMES = ['swiss-1990', 'swiss-2010', 'zurich']
    METHODS = ['concatenation_embedding',
             #'concatenation_embedding_output',
             'embedding_gate_memory',
             'embedding_gate_hidden',
             'interpolation_embedding']

    # Single Run
    SINGLE_RUN = True
    if SINGLE_RUN:
        graph_name = GRAPH_NAMES[0]
        method = METHODS[3]
        process_method(graph_name, method)
    
    # Graph Run
    GRAPH_RUN = False
    if GRAPH_RUN:
        graph_name = GRAPH_NAMES[0]
        for m in METHODS:
            process_method(graph_name, m)


    # plot graphs:
    plt.show()


                            










