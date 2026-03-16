
# TODO fix this script


import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ray.tune import ExperimentAnalysis
from sklearn.preprocessing import MinMaxScaler

from swissrivernetwork.experiment.error import Error
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.util import *
from swissrivernetwork.benchmark.attention.lstm_attention import *

from swissrivernetwork.benchmark.test_single_model import fit_column_normalizers
from swissrivernetwork.benchmark.test_isolated_station import compute_errors, plot, summary

def test_attention_model(graph_name, model):        
    df_train = read_csv_train(graph_name)
    df_train = df_train.loc[:, ~df_train.columns.isin(['epoch_day', 'has_nan'])]
    normalizers = fit_column_normalizers(df_train)

    # Read Test Data
    stations = read_stations(graph_name)
    n_stations = len(stations)
    #num_embeddings = len(stations)
    #_,edges = read_graph(graph_name)
    df = read_csv_test(graph_name)  

    # Run on validation data:
    config = {'train_split': 0.8}
    df_train = read_csv_train(graph_name)
    df_train = normalize_columns(df_train)
    df_train, df_valid = train_valid_split(config, df_train)  
    dataset_valid = STGNNSequenceFullDataset(df_valid, stations)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, shuffle=False)

    model.eval()
    validation_mse = 0
    validation_criterion = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for _,e,x,y in dataloader_valid:
            
            # only use refined output
            _,out,_ = model(e, x)            
            
            mask = ~torch.isnan(y) # mask NaNs
            loss = validation_criterion(out[mask], y[mask])
            validation_mse += loss.item()
    print(f'Validation Loss: {validation_mse:.5f}')

    # Normalize Input Values:
    for station in stations:
        df[f'{station}_at'] = normalizers[f'{station}_at'].transform(df[f'{station}_at'].values.reshape(-1, 1))
    # TODO: test if equal to column wise normalizer.. (but should)

    # Create Dataset
    dataset = STGNNSequenceFullDataset(df, stations)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    # Compute this:
    # epoch_days, prediction_norm, mask, actual, prediction
    epoch_days = [[] for _ in range(n_stations)]
    prediction_norm = [[] for _ in range(n_stations)]
    masks = [[] for _ in range(n_stations)]
    actual = [[] for _ in range(n_stations)]
    prediction = [[] for _ in range(n_stations)]    
    model.eval()
    print("How many test Sequences? -", len(dataloader))    
    weights = []
    with torch.no_grad():
        for t,e,x,y in dataloader:
                    
            # only use refined output
            _,out,w = model(e, x)            

            # Check for only one batch:
            assert 1 == out.shape[0] and 1 == y.shape[0], 'only one batch supported!'

            # Store weights:
            weights.append(w.detach().numpy())

            # Split the predictions per station:
            mask = ~torch.isnan(y)            
            for i, station in enumerate(stations):

                # Store epoch_days and prediction_norm on all days:
                epoch_days[i].append(t[0, i].detach().numpy())
                prediction_norm[i].append(out[0, i].detach().numpy()) # store normalized predictions

                # mask values:             
                mask_i = mask[0, i]                  
                masks[i].append(mask_i)                
                if mask_i.sum() == 0:
                    continue # skip if all is masked

                # Denormalize (masked output)
                ys = y[0, i][mask_i]
                outs = out[0, i][mask_i]                
                outs = normalizers[f'{station}_wt'].inverse_transform(outs.detach().numpy().reshape(-1, 1))                

                # Store values            
                actual[i].append(ys.detach().numpy()) # Original                
                prediction[i].append(outs.flatten()) # Denormalized Prediction

    # dump weights:
    weights = np.concatenate(weights, axis=0)
    np.save(f"swissrivernetwork/benchmark/dump/attention/weights-{graph_name}.npy", weights)

    # Combine arrays:
    for i in range(n_stations):
        epoch_days[i] = np.concatenate(epoch_days[i], axis=0).flatten()
        prediction_norm[i] = np.concatenate(prediction_norm[i], axis=0).flatten()
        masks[i] = np.concatenate(masks[i], axis=0).flatten()
        actual[i] = np.concatenate(actual[i], axis=0).flatten()
        prediction[i] = np.concatenate(prediction[i], axis=0).flatten()
    
    # Compute errors
    rmses = []
    maes = []
    nses = []
    ns = []
    for i,station in enumerate(stations):
        rmse, mae, nse = compute_errors(actual[i], prediction[i])
        title = summary(station, rmse, mae, nse)    
        print(title)

        # Plot Figure of Test Data
        plot(graph_name, 'attention_model_1', station, epoch_days[i][masks[i]], actual[i], prediction[i], title)

        rmses.append(rmse)
        maes.append(mae)
        nses.append(nse)
        ns.append(len(prediction))


    print('AVG RMSE:', np.mean(rmses), 'MED RMSE:', np.median(rmses))
    return rmses, maes, nses, ns

if __name__ == '__main__':

    #method = ['attention_model_1', 'graphlet_attention_model'][1]
    

#    print("Debug: top 5 stations:")
#    for idx in [21, 25, 22, 17, 9]:
#        print(f'Idx: {idx} = {stations[idx]}')
    #exit()

    # Load a model from a config:
    #analysis = ExperimentAnalysis(f'/home/benjamin/ray_results/attention_model_1-2026-03-10_17-02-58') # deprecated ?
    analysis = ExperimentAnalysis('/home/benjamin/ray_results/attention_model_1-2026-03-11_17-47-57') # good model swiss-2010

    # COPY CODE
    # Get best trial and load model:
    # This is probably not correct! (see ray evaluate)
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config    
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")
    print('use best config:', best_config)

    #graph_name = ['swiss-1990', 'swiss-2010'][1]
    graph_name = best_config['graph_name']

    # Read statistics
    stations = read_stations(graph_name)
    num_embeddings = len(stations)

    USE_TEMP_FILE = False
    if USE_TEMP_FILE:
        # Setup for GraphletAttentionModel
        best_config['embedding_size'] = 8
        best_config['hidden_size'] = 64
        best_config['num_heads'] = 2
        path = "/tmp/tmp5xikfvg6"
    else:        
        path = best_checkpoint.path

    # COPY CODE:
    # Create Model
    model_factory = ATTENTION_MODEL_FACTORY[best_config['method']]    
    model = model_factory(1, num_embeddings, best_config['embedding_size'], best_config['hidden_size'], best_config['num_heads'])
    
    #model = GraphletAttentionModel(1, num_embeddings, best_config['embedding_size'], best_config['hidden_size'], best_config['num_heads'])    
    
    model_file = sorted(os.listdir(path))[0] # single file in folder
    model.load_state_dict(torch.load(f'{path}/{model_file}'))

    test_attention_model(graph_name, model)