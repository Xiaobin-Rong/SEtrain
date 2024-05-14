import os
import torch
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
from pesq import pesq
from joblib import Parallel, delayed
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from distributed_utils import reduce_value


class Trainer:
    def __init__(self, config, model, optimizer, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, [80, 120, 150, 170, 180, 190, 200], gamma=0.5, verbose=False)
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        # training config
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

        # save the config
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
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict}

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
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    def _train_epoch(self, epoch):
        total_loss = 0
        train_bar = tqdm(self.train_dataloader, ncols=110)

        for step, (mixture, target) in enumerate(train_bar, 1):
            mixture = mixture.to(self.device)
            target = target.to(self.device)  
            
            esti_tagt = self.model(mixture)

            loss = self.loss_func(esti_tagt, target)
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_bar.postfix = 'train_loss={:.2f}'.format(total_loss / step)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()

        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_pesq_score = 0

        validation_bar = tqdm(self.validation_dataloader, ncols=123)
        for step, (mixture, target) in enumerate(validation_bar, 1):
            mixture = mixture.to(self.device)
            target = target.to(self.device)  
            
            esti_tagt = self.model(mixture)

            loss = self.loss_func(esti_tagt, target)
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            enhanced = torch.istft(esti_tagt[..., 0] + 1j*esti_tagt[..., 1], **self.config['FFT'], window=torch.hann_window(self.config['FFT']['win_length']).pow(0.5).to(self.device)).detach().cpu().numpy()
            clean = torch.istft(target[..., 0] + 1j*target[..., 1], **self.config['FFT'], window=torch.hann_window(self.config['FFT']['win_length']).pow(0.5).to(self.device)).cpu().numpy()

            pesq_score_batch = Parallel(n_jobs=-1)(
                delayed(pesq)(16000, c, e, 'wb') for c, e in zip(clean, enhanced))
            pesq_score = torch.tensor(pesq_score_batch, device=self.device).mean()
            if self.world_size > 1:
                pesq_score = reduce_value(pesq_score)
            total_pesq_score += pesq_score

            if self.rank == 0 and step <= 3:
                sf.write(os.path.join(self.sample_path,
                                    '{}_enhanced_epoch{}_pesq={:.3f}.wav'.format(step, epoch, pesq_score_batch[0])),
                                    enhanced[0], 16000)
                sf.write(os.path.join(self.sample_path,
                                    '{}_clean.wav'.format(step)),
                                    clean[0], 16000)
                
            validation_bar.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            validation_bar.postfix = 'valid_loss={:.2f}, pesq={:.4f}'.format(
                total_loss / step, total_pesq_score / step)


        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars(
                'val_loss', {'val_loss': total_loss / step, 
                             'pesq': total_pesq_score / step}, epoch)

        return total_loss / step, total_pesq_score / step


    def train(self):
        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss, score = self._validation_epoch(epoch)

            self.scheduler.step()

            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))    

            print('------------Training for {} epochs has done!------------'.format(self.epochs))
            
