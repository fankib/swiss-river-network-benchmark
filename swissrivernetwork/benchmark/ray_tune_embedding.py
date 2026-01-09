
import os
import argparse
from datetime import datetime


from ray.tune import uniform, randint, run, choice, Callback
from ray.tune.schedulers import ASHAScheduler

from swissrivernetwork.benchmark.train_embedding_model import train_lstm_embedding


'''
Run the Ray Tuner to determine best embedding architectures
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


search_space_lstm_embedding = {
    "batch_size": choice([32, 64, 128, 256]),
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01), # 10x less
    #"epochs": randint(30, 50), # 30-50    
    "epochs": 15, # default
    #"epochs": 10, # Should work fine
    #"embedding_size": randint(1, 30+1),
    "embedding_size": choice([1, 2, 5]),
    "hidden_size": randint(16, 128+1), #128  
    # num_layers is not yet implemented (= 1)  
}

def scheduler():
    return ASHAScheduler(    
        max_t = 50, # 100
        grace_period = 3,
        reduction_factor = 2)

def scheduler_long():
    return ASHAScheduler(    
        max_t = 50, # 100
        grace_period = 10,
        reduction_factor = 2)

# this one is a bit less "aggressive"
def scheduler_soft():
    return ASHAScheduler(    
        max_t = 500, # 100
        grace_period = 5,
        reduction_factor = 1.5)

def trial_name_creator(tiral):
    method = tiral.config['method']
    id = tiral.trial_id
    return f'{method}_{id}'

def run_experiment(method, graph_name, num_samples, static_embedding, shuffle_embedding, random_embedding, one_hot_embedding):

    # Each experiment has one time
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use same time for all!

    search_space = search_space_lstm_embedding.copy()            
    search_space['graph_name'] = graph_name
    search_space['method'] = method
    search_space['static_embedding'] = static_embedding
    search_space['shuffle_embedding'] = shuffle_embedding
    search_space['random_embedding'] = random_embedding
    search_space['one_hot_embedding'] = one_hot_embedding
    
    # fix embedding sizes:       
    if static_embedding:
        search_space['embedding_size'] = 3
    if one_hot_embedding:  
        from swissrivernetwork.benchmark.dataset import read_stations
        n = len(read_stations(graph_name))
        search_space['embedding_size'] = n
        print(f'[INFO]: force embedding_size={n} by one_hot_embedding')

    
    # validate inputs
    if 'vanilla' == method:
        assert search_space['static_embedding'] == False, 'No static embeddings for vanilla LSTM!'
        search_space['embedding_size'] = 0
    if shuffle_embedding:
        assert static_embedding, 'shuffle embedding is only available for static embeddings'
    if random_embedding:
        assert not shuffle_embedding, 'random is not compatible with shuffle'
        assert static_embedding, 'random embedding is only available for static embeddings'
    if one_hot_embedding:
        assert not shuffle_embedding, 'one_hot_embedding not compatible with shuffle'
        assert not random_embedding, 'one_hot_embedding not compatible with random'
        assert static_embedding, 'one_hot_embedding requires static embeddings'

    trainer = train_lstm_embedding

    analysis = run(
        trainer,        
        name = f'{method}-{now}',
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
            

if __name__ == '__main__':

    #methods = ['lstm', 'graphlet', 'lstm_embedding', 'stgnn']
    methods = ['concatenation_embedding',
             'concatenation_embedding_output',
             'embedding_gate_memory',
             'embedding_gate_hidden',
             'interpolation_embedding',
             'vanilla']
    graphs = ['swiss-1990', 'swiss-2010', 'zurich']

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', required=True, choices=methods)
    parser.add_argument('-g', '--graph', required=True, choices=graphs)
    parser.add_argument('-n', '--num_samples', required=True, type=int, help='The amount of random search samples')
    parser.add_argument('--static_embedding', action='store_true', help='Enable static embeddings (default: disabled)')
    parser.add_argument('--shuffle_embedding', action='store_true', help='Shuffle the static embeddings (default: disabeld)')
    parser.add_argument('--random_embedding', action='store_true', help='Pick random embeddings, new at each run')
    parser.add_argument('--one_hot_embedding', action='store_true', help='uses one-hot encoding for each station')
    args = parser.parse_args()

    run_experiment(args.method, args.graph, args.num_samples, args.static_embedding, args.shuffle_embedding, args.random_embedding, args.one_hot_embedding)