import os
import toml
import random
import torch
import pandas as pd
import soundfile as sf
from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(self, train_folder, shuffle, num_tot, wav_len=0, n_fft=512, hop_length=256, win_length=512):
        super().__init__()
        ### We store the noisy-clean pairs in the same folder, and use CSV file to manage all the WAV files.
        self.file_name = pd.read_csv(os.path.join(train_folder, 'INFO.csv'))['file_name'].to_list()
        
        if shuffle:
          random.seed(7)
          random.shuffle(self.file_name)
        
        if num_tot != 0:
          self.file_name = self.file_name[: num_tot]
        
        self.train_folder = train_folder
        self.wav_len = wav_len

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __getitem__(self, idx):
        noisy, fs = sf.read(os.path.join(self.train_folder, self.file_name[idx] + '_noisy.wav'), dtype="float32")
        clean, fs = sf.read(os.path.join(self.train_folder, self.file_name[idx] + '_clean.wav'), dtype="float32")

        noisy = torch.tensor(noisy)
        clean = torch.tensor(clean)

        if self.wav_len != 0:
            start = random.choice(range(len(clean) - self.wav_len * fs))
            noisy = noisy[start: start + self.wav_len*fs]
            clean = clean[start: start + self.wav_len*fs]

        noisy = torch.stft(noisy, self.n_fft, self.hop_length, self.win_length, torch.hann_window(self.win_length).pow(0.5), return_complex=False)
        clean = torch.stft(clean, self.n_fft, self.hop_length, self.win_length, torch.hann_window(self.win_length).pow(0.5), return_complex=False)

        return noisy, clean
    
    def __len__(self):
        return len(self.file_name)


if __name__=='__main__':
    from tqdm import tqdm  
    config = toml.load('config.toml')

    device = torch.device('cuda')

    train_dataset = MyDataset(**config['train_dataset'], **config['FFT'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    
    validation_dataset = MyDataset(**config['validation_dataset'], **config['FFT'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(len(train_dataloader), len(validation_dataloader))

    for noisy, clean in tqdm(train_dataloader):
        print(noisy.shape, clean.shape)
        break
        # pass

    for noisy, clean in tqdm(validation_dataloader):
        print(noisy.shape, clean.shape)
        break
 


