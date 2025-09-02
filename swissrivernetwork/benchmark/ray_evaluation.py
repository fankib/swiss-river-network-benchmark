import os
import math

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis

from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.train_isolated_station import read_stations, extract_neighbors, read_graph
from swissrivernetwork.benchmark.test_isolated_station import test_lstm, test_graphlet, test_lstm_embedding
from swissrivernetwork.benchmark.test_single_model import test_stgnn

from swissrivernetwork.gbr25.graph_exporter import plot_graph

VERBOSE = False

def experiment_analysis_isolated_station(graph_name, method, station):
    date = None
    if 'lstm' == method and 'swiss-1990' == graph_name:
        date = '05-09_19-27-00'
    if 'lstm' == method and 'swiss-2010' == graph_name:
        date = '05-13_13-32-54'
    if 'lstm' == method and 'zurich' == graph_name:
        #date = '05-27_09-41-13'
        date = '07-24_18-48-21'
    if 'graphlet' == method and 'swiss-1990' == graph_name:
        date = '05-13_16-43-09'
    if 'graphlet' == method and 'swiss-2010' == graph_name:
        date = '05-13_16-59-26'
    if 'graphlet' == method and 'zurich' == graph_name:
        #date = '07-23_11-55-42'
        date = '07-25_09-16-32'

    directory = '/home/benjamin/ray_results'
    matching_items = [item for item in os.listdir(directory) if date in item and method in item and station in item]
    assert len(matching_items) == 1, 'Identifier is not unique'
    VERBOSE and print(f'~~~ ANALYSIS for {method} at {station} ~~~')
    return ExperimentAnalysis(f'{directory}/{matching_items[0]}')

def experiment_analysis_single_model(graph_name, method):
    if 'lstm_embedding' == method and 'swiss-1990' == graph_name:
        date = '05-20_11-35-12'
    if 'lstm_embedding' == method and 'swiss-2010' == graph_name:
        date = '05-20_17-16-23'
    if 'lstm_embedding' == method and 'zurich' == graph_name:
       #date = '05-27_16-43-27' 
       date = '07-23_14-51-42' # updated stations    
    
    if 'stgnn' == method and 'swiss-1990' == graph_name:
        date = 'stgnn-2025-06-16_16-11-22'
    if 'stgnn' == method and 'swiss-2010' == graph_name:
        #date = 'stgnn-2025-06-17_18-58-54'
        date = 'stgnn-2025-06-18_18-48-43'
    if 'stgnn' == method and 'zurich' == graph_name:
        date = 'stgnn-2025-07-24_09-41-05'
    
    directory = '/home/benjamin/ray_results'
    matching_items = [item for item in os.listdir(directory) if date in item and method in item]
    assert len(matching_items) == 1, 'Identifier is not unique'
    VERBOSE and print(f'~~~ ANALYSIS for {method} ~~~')
    return ExperimentAnalysis(f'{directory}/{matching_items[0]}')

def parameter_count(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print('TOTAL MODEL PARAMETERS: ', pytorch_total_params)
    return pytorch_total_params

def evaluate_best_trial_single_model(graph_name, method):
    if 'stgnn' == method:
        analysis = experiment_analysis_single_model(graph_name, method)
    
    # Get Best Trial:
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config    
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")    
    VERBOSE and print(f'Best Trial Configuration: {best_config}')

    # Prepare Model
    if 'stgnn' == method:
        input_size = 1
        num_embeddings = len(read_stations(graph_name))
        model = SpatioTemporalEmbeddingModel(best_config['gnn_conv'], 1, num_embeddings, best_config['embedding_size'], best_config['hidden_size'], best_config['num_layers'], best_config['num_convs'])
    
    # Load Model
    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))    
    model.eval()

    # Model summary
    total_params = parameter_count(model)

    if 'stgnn' == method:
        return (*test_stgnn(graph_name, model), total_params)

def evaluate_best_trial_isolated_station(graph_name, method, station, i):    

    if 'lstm' == method or 'graphlet' == method:
        analysis = experiment_analysis_isolated_station(graph_name, method, station)
    if 'lstm_embedding' == method or 'stgnn' == method:
        analysis = experiment_analysis_single_model(graph_name, method)

    #df = analysis.dataframe()
    # Get the best Trial:
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config
    #VERBOSE and print(f"Best trial: {best_trial}")
    VERBOSE and print(f'Best Trial Configuration: {best_config}')
    
    # Get the best checkpoint
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")    

    # Determine input_size:
    if 'lstm' == method or 'lstm_embedding' == method:
        input_size = 1
    if 'graphlet' == method:
        input_size = 1+len(extract_neighbors(graph_name, station, 1))

    if 'lstm_embedding' == method:
        num_embeddings = len(read_stations(graph_name))
        embedding_size = best_config['embedding_size']

    # Create Model:        
    if 'lstm' == method or 'graphlet' == method:
        model = LstmModel(input_size, best_config['hidden_size'], best_config['num_layers'])
    if 'lstm_embedding' == method:
        model = LstmEmbeddingModel(input_size, num_embeddings, embedding_size, best_config['hidden_size'], best_config['num_layers'])
    #move
    #if 'stgnn' == method:
    #    model = SpatioTemporalEmbeddingModel(best_config['gnn_conv'], input_size, num_embeddings, embedding_size, best_config['hidden_size'], best_config['num_layers'], best_config['num_convs'])
    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))    
    model.eval()

    # model summary
    total_params = parameter_count(model)

    if 'lstm' == method:        
        return (*test_lstm(graph_name, station, model), total_params)
    if 'graphlet' == method:
        return (*test_graphlet(graph_name, station, model), total_params)
    if 'lstm_embedding' == method:
        return (*test_lstm_embedding(graph_name, station, i, model), total_params)
    # Move
    #if 'stgnn' == method:
    #    return test_stgnn(graph_name, )
    raise ValueError(f'Unknown method: {method}')

def process_method(graph_name, method):
    
    print(f'~~~ Process {method} on {graph_name} ~~~')

    failed_stations= []

    col_station = []
    col_rmse = []
    col_mae = []
    col_nse = []
    col_n = []

    # Setup
    stations = read_stations(graph_name)
    # statistics:
    print('Expected Stations: ', len(stations))

    #visualize_all_isolated_experiments(graph_name, method)
    #exit()

    total_params = 0

    if method in ['lstm', 'graphlet', 'lstm_embedding']:
        for i,station in enumerate(stations):
            if station in failed_stations:
                continue # fix this stations!
            
            # Visualize station:
            #visualize_isolated_experiment(graph_name, method, station)        
            #exit()

            #if True:
            try:            
                rmse, mae, nse, n, params = evaluate_best_trial_isolated_station(graph_name, method, station, i)
                if math.isnan(rmse):
                    failed_stations.append(station)
                    continue

                if method == 'lstm_embedding':
                    total_params = params
                else:
                    total_params += params # lstm and graphlets use different models per station.

                col_station.append(station)
                col_rmse.append(rmse)
                col_mae.append(mae)
                col_nse.append(nse)
                col_n.append(n)
            except:            
                print(f'[ERROR] Station {station} failed!')
                failed_stations.append(station)          

    if 'stgnn' == method:        
        # run model:
        rmses, maes, nses, ns, total_params = evaluate_best_trial_single_model(graph_name, method)        

        # collect per station
        for i,station in enumerate(stations):
            col_station.append(station)
            col_rmse.append(rmses[i])
            col_mae.append(maes[i])
            col_nse.append(nses[i])
            col_n.append(ns[i])
    
    print('METHOD LEARNABLE PARAMETERS:', total_params)

    print('FAILED_STATIONS:', failed_stations)

    df = pd.DataFrame(data={'Station':col_station,\
                            'RMSE':col_rmse,\
                            'MAE':col_mae,\
                            'NSE':col_nse,\
                            'N':col_n})
    
    df.to_csv(f'swissrivernetwork/journal/dump/test_results/{graph_name}_{method}.csv', index=False)
    
    plt.close('all')
    x,e = read_graph(graph_name)
    information = dict()
    color = dict()
    for station, rmse in zip(col_station, col_rmse):
        information[station] = f'{station} - (RMSE={rmse:.3f})'
        color[station] = rmse    

    # For zurich:
    plt.figure(figsize=(16,10), layout='tight')
    

    plot_graph(x,e, information=information, color=color, vmin=0.5, vmax=1.5)
    plt.savefig(f'swissrivernetwork/journal/dump/test_results/figure_{graph_name}_{method}.png', dpi=150)
    


if __name__ == '__main__':

    GRAPH_NAMES = ['swiss-1990', 'swiss-2010', 'zurich']
    METHODS = ['lstm', 'graphlet', 'lstm_embedding', 'stgnn']

    # Single Run
    SINGLE_RUN = False
    if SINGLE_RUN:
        graph_name = GRAPH_NAMES[2]
        method = METHODS[3]        
        process_method(graph_name, method)
    
    # Graph Run
    GRAPH_RUN = True
    if GRAPH_RUN:
        graph_name = GRAPH_NAMES[1]
        for m in METHODS:
            process_method(graph_name, m)


    # plot graphs:
    plt.show()


                            










