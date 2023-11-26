"""
single GPU version.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import toml
import torch

from trainer_sg import Trainer
from model import DPCRN
from datasets import MyDataset
from loss_factory import loss_wavmag, loss_mse, loss_hybrid

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

def run(config, device):

    train_dataset = MyDataset(**config['train_dataset'])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **config['train_dataloader'])
    
    validation_dataset = MyDataset(**config['validation_dataset'])
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, **config['validation_dataloader'])

    model = DPCRN(**config['network_config'])
    model.to(device)


    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])

    if config['loss']['loss_func'] == 'wav_mag':
        loss = loss_wavmag()  
    elif config['loss']['loss_func'] == 'mse':
        loss = loss_mse()
    elif config['loss']['loss_func'] == 'hybrid':
        loss = loss_hybrid()
    else:
        raise(NotImplementedError)

    trainer = Trainer(config=config, model=model,optimizer=optimizer, loss_func=loss,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      device=device)

    trainer.train()

if __name__ == '__main__':
    device = torch.device("cuda")
    config = toml.load('config.toml')
    run(config, device)

