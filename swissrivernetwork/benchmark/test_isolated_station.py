
import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ray.tune import ExperimentAnalysis
from sklearn.preprocessing import MinMaxScaler

from swissrivernetwork.experiment.error import Error
from swissrivernetwork.benchmark.model import LstmModel, LstmEmbeddingModel
from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.util import *

SHOW_PLOT = False




def run_lstm_model(model, df, normalizer_at, normalizer_wt, embedding_idx=None, use_embedding=False):
    # Predict test data:
    df['air_temperature'] = normalizer_at.transform(df['air_temperature'].values.reshape(-1, 1))
    
    # Handle NaN in air temperature:
    assert df['air_temperature'].isna().sum() == 0, 'No NaN in Input please!'
    #exit()
    #print('[DATA PREPARATION] counted NaN values in input:', df['air_temperature'].isna().sum())
    #df['air_temperature'] = df['air_temperature'].fillna(-1)

    dataset = SequenceFullDataset(df, embedding_idx)    
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    epoch_days = []
    prediction_norm = []
    masks = []
    actual = []
    prediction = []    
    with torch.no_grad():
        model.eval()
        for (t,e,x,y) in dataloader:
            if use_embedding:
                out = model(e, x)
            else:
                out = model(x)

            # Check for only one batch:
            assert 1 == out.shape[0] and 1 == y.shape[0], 'only one batch supported!'

            # Store epoch_days and prediction_norm on all days:
            epoch_days.append(t[0].detach().numpy())
            prediction_norm.append(out[0].detach().numpy()) # store normalized predictions

            # mask values:
            mask = ~torch.isnan(y)
            masks.append(mask[0])
            if mask.sum() == 0:
                continue # skip if all is masked

            y = y[mask]
            out = out[mask]
            out = normalizer_wt.inverse_transform(out.detach().numpy().reshape(-1, 1))                

            # Store values            
            actual.append(y.detach().numpy()) # original            
            prediction.append(out[:, 0])

    
    # combine arrays:
    epoch_days = np.concatenate(epoch_days, axis=0).flatten()
    prediction_norm = np.concatenate(prediction_norm, axis=0).flatten()
    masks = np.concatenate(masks, axis=0).flatten()
    actual = np.concatenate(actual, axis=0).flatten()    
    prediction = np.concatenate(prediction, axis=0).flatten()    
    return epoch_days, prediction_norm, masks, actual, prediction

def dump_predictions(graph_name, station, suffix, epoch_days, prediction):
    assert len(epoch_days) == len(prediction), 'not same amount'

    df = pd.DataFrame(data={
        'epoch_day': epoch_days,
        f'{station}_wt_hat': prediction
    })

    df.to_csv(f'swissrivernetwork/journal/dump/prediction/{graph_name}_lstm_{station}_{suffix}.csv', index=False)

def fit_normalizers(df):
    normalizer_at = MinMaxScaler().fit(df['air_temperature'].values.reshape(-1, 1))
    normalizer_wt = MinMaxScaler().fit(df['water_temperature'].values.reshape(-1, 1))
    return normalizer_at, normalizer_wt

def compute_errors(actual, prediction):
    rmse = Error.rmse(actual, prediction)
    mae = Error.mae(actual, prediction)
    nse = Error.nse(actual, prediction)
    return rmse, mae, nse

def summary(station, rmse, mae, nse):
    return f'Station {station} --\tRMSE: {rmse:.3f}\tMAE: {mae:.3f}\tNSE: {nse:.3f}'

def plot(graph_name, method, station, epoch_days, actual, prediction, title):
    plt.figure(figsize=(10,6))    
    plt.plot(epoch_days, actual, label='Actual')
    plt.plot(epoch_days, prediction, label='Prediction')
    plt.legend()
    plt.title(title.replace('\t', ' '))
    plt.savefig(f'swissrivernetwork/journal/dump/figures/{graph_name}_{method}_{station}.png', dpi=300)
    SHOW_PLOT and plt.show()

def test_graphlet(graph_name, station, model):
    # load normalizers
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)

    # prepare test data (using neighbors)
    num_hops = 1
    neighs = extract_neighbors(graph_name, station, num_hops)
    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    df_neighs = [read_csv_prediction_test(graph_name, neigh) for neigh in neighs]
    df = merge_graphlet_dfs(df, df_neighs)

    # run lstm model on it
    epoch_days, prediction_norm, mask, actual, prediction = run_lstm_model(model, df, normalizer_at, normalizer_wt)

    # comptue errors
    rmse, mae, nse = compute_errors(actual, prediction)
    title = summary(station, rmse, mae, nse)
    print(title)

    # create graphs
    plot(graph_name, 'graphlet', station, epoch_days[mask], actual, prediction, title)

    return rmse, mae, nse, len(prediction)

def test_lstm(graph_name, station, model):

    # Prepare normlalizers:
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)
    # What if we load normalizers from checkpoint?!

    # Prepare test data:
    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    epoch_days, prediction_norm, mask, actual, prediction = run_lstm_model(model, df, normalizer_at, normalizer_wt)
    dump_predictions(graph_name, station, 'test', epoch_days, prediction_norm)

    # Run on Train data as well:
    epoch_days_train, prediction_norm_train, mask_train, actual_train, prediction_train = run_lstm_model(model, df_train, normalizer_at, normalizer_wt) # do not denormalize
    dump_predictions(graph_name, station, 'train', epoch_days_train, prediction_norm_train)

    # Compute errors:
    rmse, mae, nse = compute_errors(actual, prediction)
    title = summary(station, rmse, mae, nse)    
    print(title)

    # Plot Figure of Test Data
    plot(graph_name, 'lstm', station, epoch_days[mask], actual, prediction, title)

    return rmse, mae, nse, len(prediction)

def test_lstm_embedding(graph_name, station, i, model):
    df_train = read_csv_train(graph_name)
    df_train = select_isolated_station(df_train, station)
    normalizer_at, normalizer_wt = fit_normalizers(df_train)

    df = read_csv_test(graph_name)
    df = select_isolated_station(df, station)
    epoch_days, prediction_norm, mask, actual, prediction = run_lstm_model(model, df, normalizer_at, normalizer_wt, embedding_idx=i, use_embedding=True)

    # Compute errors:
    rmse, mae, nse = compute_errors(actual, prediction)
    title = summary(station, rmse, mae, nse)    
    print(title)

    # Plot Figure of Test Data
    plot(graph_name, 'lstm_embedding', station, epoch_days[mask], actual, prediction, title)

    return rmse, mae, nse, len(prediction)



if __name__ == '__main__':

    graph_name = 'swiss-1990'
    station = '2091'
    method='lstm_embedding'

    for i,s in enumerate(read_stations(graph_name)):
        if s == station:
            break # set i

    # Select best model:
    if 'lstm' == method:
        # LSTM Model    
        analysis = ExperimentAnalysis(f'/home/benjamin/ray_results/{method}_{station}-2025-05-13_13-32-54')
        input_size = 1
    if 'graphlet' == method:
        analysis = ExperimentAnalysis(f'/home/benjamin/ray_results/{method}_{station}-2025-05-13_16-59-26')
        input_size = 1+len(extract_neighbors(graph_name, station, 1))
    if 'lstm_embedding' == method:
        analysis = ExperimentAnalysis(f'/home/benjamin/ray_results/{method}-2025-05-20_11-35-12')
        input_size = 1

    # Get best trial and load model:
    best_trial = analysis.get_best_trial(metric="validation_mse", mode="min", scope="all")
    best_config = best_trial.config    
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_mse", mode="min")

    if 'lstm_embedding' == method:
        num_embeddings = len(read_stations(graph_name))
        embedding_size = best_config['embedding_size']       

    # Create Model
    if 'lstm' == method or 'graphlet' == method:
        model = LstmModel(input_size, best_config['hidden_size'], best_config['num_layers'])
    if 'lstm_embedding' == method:
        model = LstmEmbeddingModel(input_size, num_embeddings, embedding_size, best_config['hidden_size'], best_config['num_layers'])
    model_file = sorted(os.listdir(best_checkpoint.path))[0]
    model.load_state_dict(torch.load(f'{best_checkpoint.path}/{model_file}'))    
    
    if 'lstm' == method:
        test_lstm(graph_name, station, model)
    if 'graphlet' == method:
        test_graphlet(graph_name, station, model)
    if 'lstm_embedding' == method:
        test_lstm_embedding(graph_name, station, i, model)
    plt.show()






    







