import os
import math

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ray.tune import ExperimentAnalysis

from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.lstm_embedding import EMBEDDING_MODEL_FACTORY
from swissrivernetwork.benchmark.train_isolated_station import read_stations, read_graph
from swissrivernetwork.benchmark.test_embedding_model import test_lstm_embedding

from swissrivernetwork.gbr25.graph_exporter import plot_graph

VERBOSE = False

def experiment_analysis_single_model(graph_name, method):
    # Swiss-1990
    if 'concatenation_embedding' == method and 'swiss-1990' == graph_name:
        date = '2025-09-19_14-45-03'
        #date = '2025-09-25_18-08-31' # long scheduler
    if 'concatenation_embedding_output' == method and 'swiss-1990' == graph_name:
        date = '2025-09-19_18-46-44'
    if 'embedding_gate_memory' == method and 'swiss-1990' == graph_name:
        date = '2025-09-19_20-35-57'
    if 'embedding_gate_hidden' == method and 'swiss-1990' == graph_name:
        date = '2025-09-20_00-44-55'
    if 'interpolation_embedding' == method and 'swiss-1990' == graph_name:
        date = '2025-09-20_04-58-43'
        #date = '2025-09-26_00-34-53' # long scheduler
    if 'vanilla' == method and 'swiss-1990' == graph_name:
        date = '2025-12-12_11-36-44'

    # Swiss-2010
    if 'concatenation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2025-09-23_16-42-17'    
    if 'concatenation_embedding_output' == method and 'swiss-2010' == graph_name:
        date = '2025-09-23_21-22-23'
    if 'embedding_gate_memory' == method and 'swiss-2010' == graph_name:
        date = '2025-09-23_23-38-34'
    if 'embedding_gate_hidden' == method and 'swiss-2010' == graph_name:
        date = '2025-09-24_05-46-20'
    if 'interpolation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2025-09-24_11-26-55'
    if 'vanilla' == method and 'swiss-2010' == graph_name:
        date = '2025-12-12_13-16-49'
    
    directory = '/home/benjamin/ray_results'
    matching_items = [item for item in os.listdir(directory) if date in item and method in item]
    assert len(matching_items) == 1, 'Identifier is not unique'
    VERBOSE and print(f'~~~ ANALYSIS for {method} ~~~')
    return ExperimentAnalysis(f'{directory}/{matching_items[0]}')

def experiment_analysis_lowd(graph_name, method):
    # Swiss-1990
    if 'concatenation_embedding' == method and 'swiss-1990' == graph_name:        
        date = '2025-10-06_17-39-14'
    if 'concatenation_embedding_output' == method and 'swiss-1990' == graph_name:
        date = '2025-10-06_20-25-29'
    if 'embedding_gate_memory' == method and 'swiss-1990' == graph_name:
        date = '2025-10-06_22-42-28'
    if 'embedding_gate_hidden' == method and 'swiss-1990' == graph_name:
        date = '2025-10-07_01-41-00'
    if 'interpolation_embedding' == method and 'swiss-1990' == graph_name:        
        date = '2025-10-07_04-30-35'

    # Swiss-2010
    if 'concatenation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2025-10-07_17-19-18'    
    if 'concatenation_embedding_output' == method and 'swiss-2010' == graph_name:
        date = '2025-10-07_19-40-40'
    if 'embedding_gate_memory' == method and 'swiss-2010' == graph_name:
        date = '2025-10-07_21-38-47'
    if 'embedding_gate_hidden' == method and 'swiss-2010' == graph_name:
        date = '2025-10-08_00-41-54'
    if 'interpolation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2025-10-08_03-35-30'
    
    directory = '/home/benjamin/ray_results'
    matching_items = [item for item in os.listdir(directory) if date in item and method in item]
    assert len(matching_items) == 1, 'Identifier is not unique'    
    return ExperimentAnalysis(f'{directory}/{matching_items[0]}')

def experiment_analysis_static_embedding(graph_name, method):
    # Swiss-1990
    if 'concatenation_embedding' == method and 'swiss-1990' == graph_name:        
        date = '2025-10-16_16-26-15'
    if 'concatenation_embedding_output' == method and 'swiss-1990' == graph_name:
        date = '2025-10-16_18-31-18'
    if 'embedding_gate_memory' == method and 'swiss-1990' == graph_name:
        date = '2025-10-16_20-09-44'
    if 'embedding_gate_hidden' == method and 'swiss-1990' == graph_name:
        date = '2025-10-16_22-36-49'
    if 'interpolation_embedding' == method and 'swiss-1990' == graph_name:        
        date = '2025-10-17_01-06-54'

    # Swiss-2010
    if 'concatenation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2025-10-17_05-02-59'    
    if 'concatenation_embedding_output' == method and 'swiss-2010' == graph_name:
        date = '2025-10-17_17-46-09'
    if 'embedding_gate_memory' == method and 'swiss-2010' == graph_name:
        date = '2025-10-17_20-04-08'
    if 'embedding_gate_hidden' == method and 'swiss-2010' == graph_name:
        date = '2025-10-17_23-23-57'
    if 'interpolation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2025-10-18_02-55-07'
    
    directory = '/home/benjamin/ray_results'
    matching_items = [item for item in os.listdir(directory) if date in item and method in item]
    assert len(matching_items) == 1, 'Identifier is not unique'    
    return ExperimentAnalysis(f'{directory}/{matching_items[0]}')

def experiment_analysis_shuffle_embedding(graph_name, method):
    # Swiss-1990
    if 'concatenation_embedding' == method and 'swiss-1990' == graph_name:        
        date = '2026-01-06_17-58-42'    
    if 'embedding_gate_memory' == method and 'swiss-1990' == graph_name:
        date = '2026-01-06_19-58-50'    
    if 'interpolation_embedding' == method and 'swiss-1990' == graph_name:        
        date = '2026-01-06_22-17-40'

    # Swiss-2010
    if 'concatenation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2026-01-07_02-33-41'    
    if 'embedding_gate_memory' == method and 'swiss-2010' == graph_name:
        date = '2026-01-07_05-22-15'    
    if 'interpolation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2026-01-07_08-52-03'
    
    directory = '/home/benjamin/ray_results'
    matching_items = [item for item in os.listdir(directory) if date in item and method in item]
    assert len(matching_items) == 1, 'Identifier is not unique'    
    return ExperimentAnalysis(f'{directory}/{matching_items[0]}')

def experiment_analysis_random_embedding(graph_name, method):
    # Swiss-1990
    if 'concatenation_embedding' == method and 'swiss-1990' == graph_name:        
        date = '2026-01-08_18-36-50'    
    if 'embedding_gate_memory' == method and 'swiss-1990' == graph_name:
        date = '2026-01-08_20-42-10'    
    if 'interpolation_embedding' == method and 'swiss-1990' == graph_name:        
        date = '2026-01-08_23-03-04'

    # Swiss-2010
    if 'concatenation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2026-01-09_03-12-06'    
    if 'embedding_gate_memory' == method and 'swiss-2010' == graph_name:
        date = '2026-01-09_06-03-33'    
    if 'interpolation_embedding' == method and 'swiss-2010' == graph_name:
        date = '2026-01-09_09-07-06'
    
    directory = '/home/benjamin/ray_results'
    matching_items = [item for item in os.listdir(directory) if date in item and method in item]
    assert len(matching_items) == 1, 'Identifier is not unique'    
    return ExperimentAnalysis(f'{directory}/{matching_items[0]}')


def parameter_count(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print('TOTAL MODEL PARAMETERS: ', pytorch_total_params)
    return pytorch_total_params


def evaluate_best_trial_isolated_station(graph_name, method, station, i, noise):    

    #analysis = experiment_analysis_single_model(graph_name, method)
    #analysis = experiment_analysis_lowd(graph_name, method)
    #analysis = experiment_analysis_static_embedding(graph_name, method)
    #analysis = experiment_analysis_shuffle_embedding(graph_name, method)
    analysis = experiment_analysis_random_embedding(graph_name, method)
    
    #df = analysis.dataframe()
    # Get the best Trial:    
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")    

    # Get the 100th best Trial (Hyperparameter Robustness Test):
    trials = analysis.trials
    trials.sort(key=lambda t: t.metric_analysis["validation_mse"]["min"])    
    # finished_trials.sort(key=lambda t: t.last_result["validation_mse"]) # only last reported validation_mse
    # Get the 30th best trial (index 29)
    #best_trial = trials[10] # 50ths trial    
    #assert best_trial == trials[0], 'something is off'
    
    best_config = best_trial.config
    #VERBOSE and print(f"Best trial: {best_trial}")
    VERBOSE and print(f'Best Trial Configuration: {best_config}')
    VERBOSE and print(f'Best Trial Reported mse: {best_trial.metric_analysis["validation_mse"]["min"]}')
    
    # Get the best checkpoint
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")    
    
    # Create Model:     
    num_embeddings = len(read_stations(graph_name))
    model_factory = EMBEDDING_MODEL_FACTORY[method]
    model = model_factory(1, num_embeddings, best_config['embedding_size'], best_config['hidden_size'])

    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))    
    model.eval()

    # static embedding should be stored in model file

    # Gaussian embeddings
    if noise > 0:
        embeddings = model.embedding.weight.detach().cpu().numpy()    
        std = np.std(embeddings, axis=0)    
        embeddings_noise = embeddings + np.random.normal(0, noise*std, size=embeddings.shape)    
        with torch.no_grad():
            model.embedding.weight.copy_(torch.tensor(embeddings_noise, dtype=model.embedding.weight.dtype))

    # model summary
    total_params = parameter_count(model)
        
    return (*test_lstm_embedding(graph_name, station, i, model, suffix=f'_{method}'), total_params)


def process_method(graph_name, method, noise):
    
    print(f'~~~ Process {method} on {graph_name} (Noise={noise}) ~~~')

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
    
    for i,station in enumerate(stations):
        if station in failed_stations:
            continue # fix this stations!
        
        # Visualize station:
        #visualize_isolated_experiment(graph_name, method, station)        
        #exit()

        #if True:
        try:            
            rmse, mae, nse, n, params = evaluate_best_trial_isolated_station(graph_name, method, station, i, noise)
            if math.isnan(rmse):
                failed_stations.append(station)
                continue

            total_params = params            

            col_station.append(station)
            col_rmse.append(rmse)
            col_mae.append(mae)
            col_nse.append(nse)
            col_n.append(n)
        except Exception as e:            
            print(f'[ERROR] Station {station} failed! Reason: {e}')
            failed_stations.append(station)

    
    print('METHOD LEARNABLE PARAMETERS:', total_params)

    print('FAILED_STATIONS:', failed_stations)

    df = pd.DataFrame(data={'Station':col_station,\
                            'RMSE':col_rmse,\
                            'MAE':col_mae,\
                            'NSE':col_nse,\
                            'N':col_n})
    
    df.to_csv(f'swissrivernetwork/benchmark/dump/test_results/{graph_name}_{method}.csv', index=False)
    
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
    plt.savefig(f'swissrivernetwork/benchmark/dump/test_results/figure_{graph_name}_{method}.png', dpi=150)
    
    print('RMSE:', np.mean(col_rmse), np.std(col_rmse))


if __name__ == '__main__':

    GRAPH_NAMES = ['swiss-1990', 'swiss-2010', 'zurich']
    METHODS = ['concatenation_embedding',
             'concatenation_embedding_output',
             'embedding_gate_memory',
             'embedding_gate_hidden',
             'interpolation_embedding',
             'vanilla']
    METHODS = ['concatenation_embedding', # Use for final paper
               'embedding_gate_memory',
               'interpolation_embedding']
    NOISES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

    # Single Run
    SINGLE_RUN = False
    if SINGLE_RUN:
        graph_name = GRAPH_NAMES[1]
        method = METHODS[5]    
        noise = NOISES[0]    
        process_method(graph_name, method, noise)
    
    # Graph Run
    GRAPH_RUN = True
    if GRAPH_RUN:
        graph_name = GRAPH_NAMES[1]
        noise = NOISES[0]
        for m in METHODS:
            #for noise in NOISES:
            process_method(graph_name, m, noise)

    # Print HPs:
    HP_PRINT = True
    if HP_PRINT:
        graph_name = GRAPH_NAMES[1]
        for m in METHODS:
            if m == 'vanilla':
                continue

            #analysis = experiment_analysis_single_model(graph_name, m)
            #analysis = experiment_analysis_lowd(graph_name, m)            
            #analysis = experiment_analysis_static_embedding(graph_name, m)
            #analysis = experiment_analysis_shuffle_embedding(graph_name, m)
            analysis = experiment_analysis_random_embedding(graph_name, m)

            best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")    
            best_config = best_trial.config
            print(f'hps for {m}:', best_config)
            print('~~~')

    # plot graphs:
    plt.show()


                            










