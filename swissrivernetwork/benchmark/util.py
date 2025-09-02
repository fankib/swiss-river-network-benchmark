import os
import torch
import pandas as pd

from swissrivernetwork.benchmark.dataset import read_graph
from torch_geometric.utils import k_hop_subgraph, to_undirected

def save(object, checkpoint_dir, name):
    path = os.path.join(checkpoint_dir, name)
    torch.save(object, path)

def extract_neighbors(graph_name, station, num_hops):
    # Use undirected edges:
    x, e = read_graph(graph_name)
    e = to_undirected(e)    

    # Extract k-hop neighborhood
    idx = (x[:, 2] == int(station)).nonzero().item()
    x_sub_idx, e, _, _ = k_hop_subgraph(idx, num_hops, e, relabel_nodes=True)
    x = x[x_sub_idx]
    neighs = [str(s.item()) for s in x[:, 2]]
    neighs.remove(station) # remove target station 
    return neighs

def merge_graphlet_dfs(df, df_neighs):
    for df_neigh in df_neighs:        
        df = pd.merge(df, df_neigh, on='epoch_day', how='outer')
        # fill NaN:
        col = df_neigh.columns[1]
        #df[col] = df[col].fillna(-1)
        assert df[col].isna().sum() == 0, f'NaN in neigh {col} detected!'
    has_any_nan = df.drop(columns=['air_temperature', 'water_temperature']).isna().any().any()
    assert not has_any_nan, 'There is a NaN in your data!'
    return df