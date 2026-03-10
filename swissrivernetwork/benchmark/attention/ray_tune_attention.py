
import os
import argparse
from datetime import datetime


from ray.tune import uniform, randint, run, choice, Callback
from ray.tune.schedulers import ASHAScheduler

from swissrivernetwork.benchmark.attention.train_attention_model import train_lstm_attention


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


search_space_lstm_attention = {
    "batch_size": choice([32, 64, 128]),
    "window_len": 90,
    "train_split": 0.8,
    "learning_rate": uniform(0.00001, 0.01), # 10x less    
    #"epochs": 1, # Debug
    "epochs": 15, # default    
    "embedding_size": choice([2, 5, 10]),
    "hidden_size": choice([16, 32, 64]),
    "num_heads": choice([2, 4, 8])    
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

def run_experiment(method, graph_name, num_samples):

    # Each experiment has one time
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use same time for all!

    search_space = search_space_lstm_attention.copy()            
    search_space['graph_name'] = graph_name
    search_space['method'] = method

    trainer = train_lstm_attention

    analysis = run(
        trainer,        
        name = f'{method}-{now}',
        trial_name_creator=trial_name_creator,
        config = search_space,
        scheduler = scheduler(), # ASHA is quite a speedup        
        num_samples = num_samples,
        metric = 'validation_mse',
        mode = 'min'                
    )

    print(f'\n\n~~~ Analysis of {method} ~~~')
    print('Best config: ', analysis.best_config)
            

if __name__ == '__main__':
    
    methods = ['attention_model_1']
    graphs = ['swiss-1990', 'swiss-2010', 'zurich']

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', required=True, choices=methods)
    parser.add_argument('-g', '--graph', required=True, choices=graphs)
    parser.add_argument('-n', '--num_samples', required=True, type=int, help='The amount of random search samples')    
    args = parser.parse_args()

    run_experiment(args.method, args.graph, args.num_samples)