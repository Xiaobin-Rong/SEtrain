"""
multiple GPUs version, using DDP training.
"""
import os
import torch
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from pesq import pesq
from distributed_utils import reduce_value


class Trainer:
    def __init__(self, config, model, optimizer, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.model = model

        self.optimizer = optimizer
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=5,verbose=True)
                
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device

        ## training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
            
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        ## save the config
        if self.rank == 0:
            with open(
                os.path.join(
                    self.exp_path, 'config.toml'.format(datetime.now().strftime("%Y-%m-%d-%Hh%Mm"))), 'w') as f:

                toml.dump(config, f)

            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 0

        if self.resume:
            self._resume_checkpoint()

        self.sr = config['listener']['listener_sr']

        self.loss_func = self.loss_func.to(self.device)


    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'model': self.model.module.state_dict()}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(4)}.tar'))

        if score > self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.module.load_state_dict(checkpoint['model'])

    def _train_epoch(self, epoch):
        total_loss = 0
        self.train_dataloader = tqdm(self.train_dataloader, ncols=110)

        for step, (mixture, target) in enumerate(self.train_dataloader, 1):
            mixture = mixture.to(self.device)
            target = target.to(self.device)  

            esti_tagt = self.model(mixture)

            loss = self.loss_func(esti_tagt, target)
            loss = reduce_value(loss)
            total_loss += loss.item()

            self.train_dataloader.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_dataloader.postfix = 'loss={:.3f}'.format(total_loss / step)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()

        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_pesq_score = 0

        self.validation_dataloader = tqdm(self.validation_dataloader, ncols=132)
        for step, (mixture, target) in enumerate(self.validation_dataloader, 1):
            mixture = mixture.to(self.device)
            target = target.to(self.device)  
            
            esti_tagt = self.model(mixture)

            loss = self.loss_func(esti_tagt, target)
            loss = reduce_value(loss)
            total_loss += loss.item()

            enhanced = torch.istft(esti_tagt[..., 0] + 1j*esti_tagt[..., 1], **self.config['FFT'], torch.hann_window(self.config['FFT']['win_length']).pow(0.5).to(self.device))
            clean = torch.istft(target[..., 0] + 1j*target[..., 1], **self.config['FFT'], torch.hann_window(self.config['FFT']['win_length']).pow(0.5).to(self.device))

            enhanced = enhanced.squeeze().cpu().numpy()
            clean = clean.squeeze().cpu().numpy()

            pesq_score = pesq(16000, clean, enhanced, 'wb')
            pesq_score = reduce_value(torch.tensor(pesq_score, device=self.device))
            total_pesq_score += pesq_score
            
            if self.args==0 and step <= 3:
                sf.write(os.path.join(self.sample_path,
                                    '{}_enhanced_epoch{}_pesq={:.3f}.wav'.format(step, epoch, pesq_score)),
                                    enhanced, 16000)
                sf.write(os.path.join(self.sample_path,
                                    '{}_clean.wav'.format(step)),
                                    clean, 16000)
                
            self.validation_dataloader.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_dataloader.postfix = 'loss={:.2f}, pesq={:.4f}'.format(
                total_loss / step, total_pesq_score / step)


        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars(
                'val_loss', {'val_loss': total_loss / step, 
                             'pesq': total_pesq_score / step}, epoch)

        return total_loss / step, total_pesq_score / step


    def train(self):
        if self.rank == 0:
            timestamp_txt = os.path.join(self.exp_path, 'timestamp.txt')
            mode = 'a' if os.path.exists(timestamp_txt) else 'w'
            with open(timestamp_txt, mode) as f:
                f.write('[{}] start for {} epochs\n'.format(
                    datetime.now().strftime("%Y-%m-%d-%H:%M"), self.epochs))

        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss, score = self._validation_epoch(epoch)

            self.scheduler.step(valid_loss)

            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))    

            print('------------Training for {} epochs has done!------------'.format(self.epochs))

            with open(timestamp_txt, 'a') as f:
                f.write('[{}] end\n'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))

        