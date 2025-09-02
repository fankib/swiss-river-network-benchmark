
import os
import argparse
from datetime import datetime


from ray.tune import uniform, randint, run, choice, Callback
from ray.tune.schedulers import ASHAScheduler

from swissrivernetwork.benchmark.train_isolated_station import train_lstm, read_stations, train_graphlet
from swissrivernetwork.benchmark.train_single_model import train_lstm_embedding, train_stgnn


'''
Run the Ray Tuner to determine best architectures
'''


class MemoryLimitCallback(Callback):
    def __init__(self, mem_limit_mb):
        self.mem_limit_mb = mem_limit_mb

    def on_trial_result(self, iteration, trials, trial, result, **info):
        import psutil
        process = psutil.Process(trial.runner.pid)
        mem = process.memory_info().rss / (1024 ** 2)
        if mem > self.mem_limit_mb:
            print(f"Stopping trial {trial.trial_id} due to high memory usage: {mem} MB")
            trial.stop()


# Use actual working dir
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

search_space_lstm = {
    "batch_size": randint(32, 256+1),
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.0001, 0.01),
    #"epochs": randint(30, 50), # 30-50    
    "epochs": 30,
    "hidden_size": randint(16, 128+1), #128    
    "num_layers": randint(1,3+1) # more layers!
}

search_space_lstm_embedding = {
    "batch_size": randint(32, 256+1),
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01), # 10x less
    #"epochs": randint(30, 50), # 30-50    
    "epochs": 30,
    "embedding_size": randint(1, 30+1),
    "hidden_size": randint(16, 128+1), #128    
    "num_layers": randint(1,3+1) # more layers!
}

search_space_stgnn = {
    "batch_size": randint(1, 10), # Batch size is times |Nodes| 30-50 bigger
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01), # 10x less
    #"epochs": randint(30, 50), # 30-50    
    "epochs": 30,
    "embedding_size": randint(1, 30+1),
    "hidden_size": randint(16, 128+1), #128    
    "num_layers": randint(1,3+1), # more layers!
    #"gnn_conv": choice(['GCN', 'GIN']),
    "gnn_conv": choice(['GraphSAGE']),
    "num_convs": randint(1, 7+1)
}

def scheduler():
    return ASHAScheduler(    
        max_t = 200, # 100
        grace_period = 3,
        reduction_factor = 2)

# this one is a bit less "aggressive"
def scheduler_soft():
    return ASHAScheduler(    
        max_t = 200, # 100
        grace_period = 5,
        reduction_factor = 1.5)

def scheduler_single_model_soft():
    return ASHAScheduler(    
        max_t = 500, # 100
        grace_period = 5,
        reduction_factor = 1.5)

def scheduler_single_model_hard():
    return ASHAScheduler(    
        max_t = 500,
        grace_period = 3,
        reduction_factor = 2)

def run_experiment(method, graph_name, num_samples):

    # Each experiment has one time
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use same time for all!

    # update search space (!)
    if 'lstm' == method or 'graphlet' == method:
        
        for station in read_stations(graph_name):

            search_space = search_space_lstm.copy()
            search_space['station'] = station
            search_space['graph_name'] = graph_name                       

            trainer = None
            if 'lstm' == method:
                trainer = train_lstm
            if 'graphlet' == method:
                trainer = train_graphlet

            analysis = run(
                trainer,        
                name = f'{method}_{station}-{now}',
                config = search_space,
                #scheduler = scheduler_soft(), # ASHA is quite a speedup
                scheduler = scheduler(),
                num_samples = num_samples,
                metric = 'validation_mse',
                mode = 'min'
            )

            print(f'\n\n~~~ Analysis of {method} ~~~')
            print('Best config: ', analysis.best_config)
    
    if 'lstm_embedding' == method:
        search_space = search_space_lstm_embedding.copy()            
        search_space['graph_name'] = graph_name                       

        trainer = train_lstm_embedding

        analysis = run(
            trainer,        
            name = f'{method}-{now}',
            config = search_space,
            scheduler = scheduler_soft(), # ASHA is quite a speedup
            num_samples = num_samples,
            metric = 'validation_mse',
            mode = 'min'                
        )

    if 'stgnn' == method:
        search_space = search_space_stgnn.copy()            
        search_space['graph_name'] = graph_name

        # add GAT Heads
        search_space['num_heads'] = 0
        if search_space['gnn_conv'] == 'GAT':
            search_space['num_heads'] = randint(1, 8)            

        trainer = train_stgnn

        analysis = run(
            trainer,        
            name = f'{method}-{now}',
            config = search_space,
            scheduler = scheduler_single_model_hard(), # ASHA is quite a speedup
            num_samples = num_samples,
            metric = 'validation_mse',
            mode = 'min',
            max_concurrent_trials=20 # Fix memory issues            
        )
    
    if method in ['lstm_embedding', 'stgnn']:
        print(f'\n\n~~~ Analysis of {method} ~~~')
        print('Best config: ', analysis.best_config)
            

if __name__ == '__main__':

    methods = ['lstm', 'graphlet', 'lstm_embedding', 'stgnn']
    graphs = ['swiss-1990', 'swiss-2010', 'zurich']

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', required=True, choices=methods)
    parser.add_argument('-g', '--graph', required=True, choices=graphs)
    parser.add_argument('-n', '--num_samples', required=True, type=int, help='The amount of random search samples')
    args = parser.parse_args()

    run_experiment(args.method, args.graph, args.num_samples)