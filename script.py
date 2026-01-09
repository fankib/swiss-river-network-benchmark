import torch
from swissrivernetwork.benchmark.dataset import *

for graph_name in ['swiss-1990', 'swiss-2010']:
    stations = read_stations(graph_name)
    n = len(stations)
    perm = torch.randperm(n)
    print(f'~~~ {graph_name} ~~~')
    print(perm)