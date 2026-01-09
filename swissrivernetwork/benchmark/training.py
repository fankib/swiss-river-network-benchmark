
import torch
import tempfile
import torch.nn as nn

from ray.train import Checkpoint, report

from swissrivernetwork.benchmark.util import save

def training_loop(config, dataloader_train, dataloader_valid, model, n_valid, use_embedding, edges=None):

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
                if edges is not None:
                    out = model(x, edges)
                elif use_embedding:
                    out = model(e, x)
                else:
                    out = model(x)
                mask = ~torch.isnan(y) # mask NaNs            
                loss = criterion(out[mask], y[mask])
                loss.backward()
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