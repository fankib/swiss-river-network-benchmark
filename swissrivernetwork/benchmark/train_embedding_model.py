
import torch
import torch.nn.functional as F

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.training import training_loop
from swissrivernetwork.benchmark.lstm_embedding import EMBEDDING_MODEL_FACTORY

import rasterio
from pyproj import Transformer

def train_lstm_embedding(config):

    # Setup Dataset
    graph_name = config['graph_name']    
    stations = read_stations(graph_name)
    num_embeddings = len(stations)

    df = read_csv_train(graph_name)
    datasets_train = []
    datasets_valid = []
    for i,station in enumerate(stations):
        df_station = select_isolated_station(df, station)
        dataset_train,dataset_valid = create_dataset_embedding(config, df_station, i)
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid)

    # create model:
    model_factory = EMBEDDING_MODEL_FACTORY[config['method']]
    model = model_factory(1, num_embeddings, config['embedding_size'], config['hidden_size'])

    # test for static embedding:
    if config['static_embedding']:
        assert config['embedding_size'] == 3 or config['one_hot_embedding'], 'static embedding require embeddings_size=3'
        insert_static_embedding(graph_name, model, config['shuffle_embedding'], config['random_embedding'], config['one_hot_embedding'])

    # Run Training Loop!
    training_loop(config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=True)
    

def insert_static_embedding(graph_name, model, shuffle, random, one_hot):
    tile_path = "/home/benjamin/git/airstart3d/data/airstart3d/elevation/eudem_dem_32632_switzerland.tif"
    with rasterio.open(tile_path) as elevation_raster:
        x,_ = read_graph(graph_name)
        elevations = []
        for ch_x, ch_y in x[:, 0:2]:
            elevation = get_evaluation(ch_x, ch_y, elevation_raster)
            elevations.append(elevation)

    # h-stack elevation to x..
    embeddings = torch.cat((x[:, 0:2], torch.tensor(elevations).unsqueeze(1)), dim=1)
    # normalize
    min_vals = embeddings.min(dim=0, keepdim=True).values
    max_vals = embeddings.max(dim=0, keepdim=True).values
    embeddings_norm = (embeddings - min_vals) / (max_vals - min_vals + 1e-8)

    # shuffle embeddings (change to random permutation):
    if shuffle:
        if graph_name == 'swiss-1990':
            permutation = torch.tensor([20,  8, 12, 22,  0,  4,  9, 23, 25,  1,  3, 26, 24, 16,  7, 10, 14,  6, 5, 11, 21, 17, 27, 18, 15, 19,  2, 13])
        if graph_name == 'swiss-2010':
            permutation = torch.tensor([32,  2, 47, 29, 55,  6, 28, 58, 10, 21, 20, 45,  5, 61, 57,  0, 11,  1, 8, 13, 15, 44, 42, 51, 49, 17, 18, 34, 30, 62, 50, 25, 56, 59, 26, 60, 7, 36, 40, 16, 22, 46, 37,  9, 33,  4,  3, 53, 39, 12, 27, 54, 19, 41, 24, 35, 23, 43, 14, 52, 38, 31, 48])
        assert graph_name != 'zurich', 'Zurich is not supported'
        embeddings_norm = embeddings_norm[permutation] # apply permutation
        print('[INFO] static embeddings shuffled!')

    if random:
        embeddings_norm = torch.rand_like(embeddings_norm)
        print('[INFO] pick random embeddings!')
    
    if one_hot:
        idx = torch.arange(embeddings_norm.size(0))
        embeddings_norm = F.one_hot(idx)
        print('[INFO] one hot encoded embeddings')

    # update embeddings:    
    model.embedding = nn.Embedding.from_pretrained(embeddings_norm, freeze=True)
    print("[INFO] static embedding installed")    


def get_evaluation(x, y, elevation_raster): #ch1903+ to utm32 to elevation    
    # CH1903 (21781) (LV05 is 2056) UTM Zone 32N
    transformer = Transformer.from_crs("EPSG:21781", "EPSG:32632", always_xy=True)
    # Umwandlung in UTM (Meter)
    east, north = transformer.transform(x, y)
    row, col = elevation_raster.index(east, north)
    value = elevation_raster.read(1)[row, col]
    #print(f"Elevation at ({x}, {y}): {value} m")
    return value

def create_dataset_embedding(config, df, i):
    # Normalize
    df = normalize_isolated_station(df)   

    # Train/Validation split
    df_train, df_valid = train_valid_split(config, df)

    # Create datasets
    dataset_train = SequenceWindowedDataset(config['window_len'], df_train, embedding_idx=i)
    dataset_valid = SequenceFullDataset(df_valid, embedding_idx=i)
    return dataset_train, dataset_valid




if __name__ == '__main__':

    # fix 2010 bug:
    #graph_name = 'swiss-2010'
    graph_name = 'swiss-1990'    

    # model:
    method = ['concatenation_embedding',
             'concatenation_embedding_output',
             'embedding_gate_memory',
             'embedding_gate_hidden',
             'interpolation_embedding',
             'vanilla'][0]

    # read stations:
    print(read_stations(graph_name))

    config = {        
        'graph_name': graph_name,
        'method': method,
        'batch_size': 256,
        'window_len': 90,
        'train_split': 0.8,
        'learning_rate': 0.001,
        'epochs': 15,
        #'embedding_size': 3,
        'embedding_size': 28, #one_hot
        'hidden_size': 32,   
        'static_embedding': True,     
        'shuffle_embedding': False,
        'random_embedding': False,
        'one_hot_embedding': True
    }
        
    train_lstm_embedding(config)    
