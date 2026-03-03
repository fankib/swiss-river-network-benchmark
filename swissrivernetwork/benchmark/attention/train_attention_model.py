
import torch
import torch.nn.functional as F

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.attention.lstm_attention import ATTENTION_MODEL_FACTORY

import rasterio
from pyproj import Transformer

def train_lstm_attention(config):

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
    model_factory = ATTENTION_MODEL_FACTORY[config['method']]
    model = model_factory(1, num_embeddings, config['embedding_size'], config['hidden_size'])

    # Run Training Loop!
    training_loop(config, dataloader_train, dataloader_valid, model, len(dataset_valid), use_embedding=True)   

# This training loop consists of two losses
# Loss1: direct prediction loss (LSTM-E Model)
# Loss2: 
def training_loop(config, dataloader_train, dataloader_valid, model):

    try:
        # Run the Trainig loop on the Model
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()
        validation_criterion = nn.MSELoss(reduction='mean') # weight all samples equally
        
        for epoch in range(config['epochs']):
            model.train()
            losses = []
            for _,e,x,y in dataloader_train:
                optimizer.zero_grad()

                # embedding by default                
                out1, out2 = model(e, x) # Two losses
                
                mask = ~torch.isnan(y) # mask NaNs            
                loss1 = criterion(out1[mask], y[mask])
                loss1.backward()

                # refined loss:
                loss2 = criterion(out2[mask], y[mask])
                loss2.backward()

                optimizer.step()
                losses.append(loss.item())

            model.eval()
            validation_mse = 0
            with torch.no_grad():
                for _,e,x,y in dataloader_valid:
                    if edges is not None:
                        out = model(x, edges)
                    elif use_embedding:
                        out = model(e, x)
                    else:
                        out = model(x)
                    mask = ~torch.isnan(y) # mask NaNs
                    loss = validation_criterion(out[mask], y[mask])
                    validation_mse += loss.item()
            # use mean reducer -- is not perfect but makes more sense #validation_mse /= n_valid # normalize by dataset length

            # Register Ray Checkpoint
            checkpoint_dir = tempfile.mkdtemp()
            save(model.state_dict(), checkpoint_dir, f'lstm_epoch_{epoch+1}.pth')
            #save(normalizer_at, checkpoint_dir, 'normalizer_at.pth')
            #save(normalizer_wt, checkpoint_dir, 'normalizer_wt.pth')
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # report epoch loss
            report({"validation_mse": validation_mse}, checkpoint=checkpoint)        
            print(f'End of Epoch {epoch+1}: {validation_mse:.5f}')
            # Debug for static embedding:
            #print('embedding after:', model.embedding.weight)
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            report(done=True, status="OOM")        
        else:
            raise


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
    method = ['attention_model_1',
              'attention_model_2'][2]

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
