
import os
import argparse
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import tempfile


#### Ray Tuner ###
from ray.tune import uniform, randint, run, choice, Callback
from ray.tune.schedulers import ASHAScheduler

# Use actual working dir
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

search_space_traffic = {
    "batch_size": choice([128, 256, 512]),
    "window_len": 96,
    "pred_len": 24,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01), # 10x less    
    #"epochs": 1, # DEBUG
    "epochs": 15, # default    
    "embedding_size": randint(1, 30+1),    
    "hidden_size": randint(16, 256+1), #128   
    # Stacking?   
}

def scheduler():
    return ASHAScheduler(    
        max_t = 50, # 100
        grace_period = 3,
        reduction_factor = 2)

def trial_name_creator(tiral):
    method = tiral.config['method']
    id = tiral.trial_id
    return f'{method}_{id}'

def run_experiment(method, num_samples):

    # Each experiment has one time
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use same time for all!

    search_space = search_space_traffic.copy()                
    search_space['method'] = method    

    trainer = train_lstm_traffic

    analysis = run(
        trainer,        
        name = f'traffic_{method}-{now}',
        trial_name_creator=trial_name_creator,
        config = search_space,
        scheduler = scheduler(), # ASHA is quite a speedup
        #scheduler = scheduler_long(),
        num_samples = num_samples,
        metric = 'validation_mse',
        mode = 'min'                
    )

    print(f'\n\n~~~ Analysis of {method} ~~~')
    print('Best config: ', analysis.best_config)
            



### Trainer ###

from swissrivernetwork.benchmark.dataset import select_isolated_station, train_valid_split, SequenceWindowedDataset, SequenceFullDataset
from swissrivernetwork.benchmark.traffic_dataset.lstm_traffic import EMBEDDING_MODEL_FACTORY

class TrafficSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, df, window_len, pred_len, jump_len, embedding_idx):
        series = df.values
        self.series = (series - series.mean()) / series.std() # standardize

        self.window_len = window_len
        self.pred_len = pred_len
        self.jump_len = jump_len
        self.embedding_idx = embedding_idx

        self.n_samples = len(self.series) - window_len - pred_len + 1

    def __len__(self):
        return self.n_samples // self.jump_len
    
    def __getitem__(self, idx):
        idx = idx * self.jump_len
        x = self.series[idx : idx + self.window_len]
        y = self.series[idx + self.window_len : idx + self.window_len + self.pred_len]

        x = torch.FloatTensor(x).unsqueeze(-1)
        y = torch.FloatTensor(y)        
        embs = torch.LongTensor([self.embedding_idx]*x.shape[0])

        return embs, x, y    

def read_traffic_csv():
    return pd.read_csv("../data/iTransformer_datasets/traffic/traffic.csv")

def read_traffic_stations():
    df = read_traffic_csv()
    return df.columns[1:-1]

def create_dataset_embedding(config, df, i):
    # Data is normalized

    # Train/Validation split
    df_train, df_valid = train_valid_split(config, df)

    # Create datasets
    dataset_train = TrafficSequenceDataset(df_train, config['window_len'], config['pred_len'], 1, i)
    dataset_valid = TrafficSequenceDataset(df_valid, config['window_len'], config['pred_len'], config['pred_len'], i)
    return dataset_train, dataset_valid

def train_lstm_traffic(config):
    stations = read_traffic_stations()
    num_embeddings = len(stations)

    df = read_traffic_csv() # not yet a train set

    datasets_train = []
    datasets_valid = []
    #for i,station in enumerate(stations[:5]): # DEBUG!!!
    for i,station in enumerate(stations):
        df_station = df[station]
        dataset_train,dataset_valid = create_dataset_embedding(config, df_station, i)
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataset_valid = torch.utils.data.ConcatDataset(datasets_valid)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid)

     # create model:
    model_factory = EMBEDDING_MODEL_FACTORY[config['method']]
    model = model_factory(1, config['pred_len'], num_embeddings, config['embedding_size'], config['hidden_size'])

    # Run Training Loop!
    training_loop(config, dataloader_train, dataloader_valid, model)


### The Training Loop ###
from ray.train import Checkpoint, report

from swissrivernetwork.benchmark.util import save

def training_loop(config, dataloader_train, dataloader_valid, model):
    
    # Run the Trainig loop on the Model
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    validation_criterion = nn.MSELoss(reduction='mean') # weight all samples equally
        
    for epoch in range(config['epochs']):
        model.train()
        losses = []
        for e,x,y in dataloader_train:
            optimizer.zero_grad()                
            out = model(e, x)
                
            mask = ~torch.isnan(y) # mask NaNs ? TODO: nans == 0?!        
            loss = criterion(out[mask], y[mask])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            #print('loop done!')

        model.eval()
        validation_mse = 0
        with torch.no_grad():
            for e,x,y in dataloader_valid:                    
                out = model(e, x)
                    
                mask = ~torch.isnan(y) # mask NaNs
                loss = validation_criterion(out[mask], y[mask]) # denormalize!
                validation_mse += loss.item()
            validation_mse /= len(dataloader_valid) # normalize by amount of batches

            # Register Ray Checkpoint
            checkpoint_dir = tempfile.mkdtemp()
            save(model.state_dict(), checkpoint_dir, f'lstm_epoch_{epoch+1}.pth')            
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # report epoch loss
            report({"validation_mse": validation_mse}, checkpoint=checkpoint)        
            print(f'End of Epoch {epoch+1}: {validation_mse:.5f}')            


### Main RUNNER ###

def debug():
    #print(read_traffic_stations())
    config = {
        "method": 'concatenation_embedding',
        "batch_size": 32,
        "window_len": 96,
        "pred_len": 96,
        "train_split": 0.8,
        "learning_rate": 0.01,
        "epochs": 15, # default    
        "embedding_size": 13,    
        "hidden_size": 128    
    }
    #train_lstm_traffic(config)
    pass

if __name__ == '__main__':

    # Debug
    #debug()
    #exit()
    # End of Debug
    
    methods = ['concatenation_embedding',
             'concatenation_embedding_output',
             'embedding_gate_memory',
             'embedding_gate_hidden',
             'interpolation_embedding']    

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', required=True, choices=methods)    
    parser.add_argument('-n', '--num_samples', required=True, type=int, help='The amount of random search samples')    
    args = parser.parse_args()

    run_experiment(args.method, args.num_samples)