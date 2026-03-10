
import torch
import tempfile
import torch.nn.functional as F
from ray.train import Checkpoint, report

from swissrivernetwork.benchmark.dataset import *
from swissrivernetwork.benchmark.model import *
from swissrivernetwork.benchmark.attention.lstm_attention import ATTENTION_MODEL_FACTORY

from swissrivernetwork.benchmark.util import save


def train_lstm_attention(config):
    # Setup Dataset
    graph_name = config['graph_name']    
    stations = read_stations(graph_name)
    num_embeddings = len(stations)

    # Read and prepare data
    df = read_csv_train(graph_name)
    df = normalize_columns(df)

    # Create Datasets
    df_train, df_valid = train_valid_split(config, df)
    dataset_train = STGNNSequenceWindowedDataset(config['window_len'], df_train, stations)
    dataset_valid = STGNNSequenceFullDataset(df_valid, stations)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid)

    # create model:
    model_factory = ATTENTION_MODEL_FACTORY[config['method']]
    model = model_factory(1, num_embeddings, config['embedding_size'], config['hidden_size'], config['num_heads'])    

    # Run Training Loop!
    training_loop(config, dataloader_train, dataloader_valid, model)   



# This training loop consists of two losses
# Loss1: direct prediction loss (LSTM-E Model)
# Loss2: refined loss 
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
                loss1.backward(retain_graph=True)

                # refined loss:
                loss2 = criterion(out2[mask], y[mask])
                loss2.backward()

                optimizer.step()
                losses.append(loss1.item() + loss2.item())

            model.eval()
            validation_mse = 0
            with torch.no_grad():
                for _,e,x,y in dataloader_valid:
                    
                    # only use refined output
                    _,out = model(e, x)
                    
                    mask = ~torch.isnan(y) # mask NaNs
                    loss = validation_criterion(out[mask], y[mask])
                    validation_mse += loss.item()            

            # Register Ray Checkpoint
            checkpoint_dir = tempfile.mkdtemp()
            save(model.state_dict(), checkpoint_dir, f'lstm_epoch_{epoch+1}.pth')            
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # report epoch loss
            report({"validation_mse": validation_mse}, checkpoint=checkpoint)        
            print(f'End of Epoch {epoch+1}: {validation_mse:.5f}')            
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            report(done=True, status="OOM")        
        else:
            raise




if __name__ == '__main__':
    
    #graph_name = 'swiss-2010'
    graph_name = 'swiss-1990'

    # model:
    method = ['attention_model_1',
              'attention_model_2'][0]

    # read stations:
    print(read_stations(graph_name))

    config = {
        'graph_name': graph_name,
        'method': method,
        'batch_size': 64,
        'window_len': 90,
        'train_split': 0.8,
        'learning_rate': 0.0079,
        'epochs': 15,
        'embedding_size': 2,      
        'hidden_size': 32,    
        'num_heads': 4,
    }    
        
    train_lstm_attention(config)    
