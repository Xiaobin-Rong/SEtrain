"""
multiple GPUs version, using DDP training.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
import toml
import torch
import argparse
import torch.distributed as dist

from trainer import Trainer
from model import DPCRN
from datasets import MyDataset
from loss_factory import loss_wavmag, loss_mse, loss_hybrid

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def run(rank, config, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)

    train_dataset = MyDataset(**config['train_dataset'], **config['FFT'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                                    **config['train_dataloader'])
    
    validation_dataset = MyDataset(**config['validation_dataset'], **config['FFT'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, sampler=validation_sampler,
                                                        **config['validation_dataloader'])

    model = DPCRN(**config['network_config'])
    model.to(args.device)

    # convert to DDP model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
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
                      train_sampler=train_sampler, args=args)

    trainer.train()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='config.toml')

    args = parser.parse_args()

    config = toml.load(args.config)
    args.world_size = config['DDP']['world_size']
    torch.multiprocessing.spawn(
        run, args=(config, args,), nprocs=args.world_size, join=True)

