import toml
import torch
import torch.nn as nn
from torch_stoi import NegSTOILoss


config = toml.load('config.toml')


class loss_mse(nn.Module):
    def __init__(self):
        super(loss_mse, self).__init__()
        self.window = torch.hann_window(config['FFT']['win_length']).pow(0.5)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, est, clean):
        """ inputs: spectrograms, (B,F,T,2) """
        data_len = min(est.shape[-1], clean.shape[-1])
        est = est[..., :data_len]
        clean = clean[..., :data_len]

        est_stft = torch.stft(est, **config['FFT'], center=True, window=self.window.to(est.device), return_complex=False)   
        clean_stft = torch.stft(clean, **config['FFT'], center=True, window=self.window.to(clean.device), return_complex=False)
        est_stft_real, est_stft_imag = est_stft[:,:,:,0], est_stft[:,:,:,1]
        clean_stft_real, clean_stft_imag = clean_stft[:,:,:,0], clean_stft[:,:,:,1]
        est_mag = torch.sqrt(est_stft_real**2 + est_stft_imag**2 + 1e-12)
        clean_mag = torch.sqrt(clean_stft_real**2 + clean_stft_imag**2 + 1e-12)
        est_real_c = est_stft_real / (est_mag**(0.7))
        est_imag_c = est_stft_imag / (est_mag**(0.7))
        clean_real_c = clean_stft_real / (clean_mag**(0.7))
        clean_imag_c = clean_stft_imag / (clean_mag**(0.7))

        loss = 0.7 * self.mse_loss(est_mag**(0.3), clean_mag**(0.3)) + \
                0.3 * (self.mse_loss(est_real_c, clean_real_c) + \
                self.mse_loss(est_imag_c, clean_imag_c))
        
        return loss


class loss_sisnr(nn.Module):
    def __init__(self):
        super(loss_sisnr, self).__init__()

    def forward(self, est, clean):
        """ inputs: waveform, (B,...,T) """
        data_len = min(est.shape[-1], clean.shape[-1])
        est = est[..., :data_len]
        clean = clean[...,:data_len]
        est = est - torch.mean(est, dim=-1, keepdim=True)
        clean = clean - torch.mean(clean, dim=-1, keepdim=True)

        target = torch.sum(est * clean, 1, keepdim=True) * clean / \
            torch.sum(clean**2 + 1e-8, 1, keepdim=True)
        noise = est - target
        sisnr = 10*torch.log10((torch.sum(target**2, 1) + 1e-8)/(torch.sum(noise**2, 1) + 1e-8))
        est_std = torch.std(est, dim=1)
        clean_std = torch.std(clean, dim=1)
        
        com_factor = torch.minimum((est_std + 1e-8) / (clean_std + 1e-8),
                                   (clean_std + 1e-8) / (est_std + 1e-8))

        return -torch.mean(sisnr * com_factor)


class loss_stoi(torch.nn.Module):
    def __init__(self, sample_rate):
        super(loss_stoi, self).__init__()
        self.NegSTOI = NegSTOILoss(sample_rate=sample_rate)

    def forward(self, est, clean):
        """ inputs: waveform, (B,...,T) """
        data_len = min(est.shape[-1], clean.shape[-1])
        est = est[..., : data_len]
        clean = clean[...,: data_len]

        return self.NegSTOI(est, clean).mean()


class loss_wavmag(nn.Module):
    def __init__(self):
        super(loss_wavmag, self).__init__()

    def forward(self, est_stft, clean_stft, alpha=10):
        """ inputs: spectrograms, (B,F,T,2) """
        device = est_stft.device

        est_stft = est_stft[..., 0] + 1j*est_stft[..., 1]
        clean_stft = clean_stft[..., 0] + 1j*clean_stft[..., 1]

        estimated = torch.istft(est_stft, **config['FFT'], window=torch.hann_window(512).pow(0.5).to(device))
        clean = torch.istft(clean_stft, **config['FFT'], window=torch.hann_window(512).pow(0.5).to(device))
        
        loss_wav = torch.norm((estimated - clean), p=1) / clean.numel() * 100
        loss_mag = torch.norm(abs(est_stft) - abs(clean_stft), p=1) / clean_stft.numel() * 100
        return alpha*loss_wav + loss_mag
    
    
class loss_hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.window = torch.hann_window(config['FFT']['win_length']).pow(0.5)

    def forward(self, pred_stft, true_stft):
        """ inputs: spectrograms, (B,F,T,2) """
        device = pred_stft.device

        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = torch.mean((pred_real_c - true_real_c)**2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c)**2)
        mag_loss = torch.mean((pred_mag**(0.3)-true_mag**(0.3))**2)
        

        y_pred = torch.istft(pred_stft_real+1j*pred_stft_imag, **config['FFT'], window=self.window.to(device))
        y_true = torch.istft(true_stft_real+1j*true_stft_imag, **config['FFT'], window=self.window.to(device))
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)
        sisnr =  - torch.log10(torch.sum(torch.square(y_true),dim=-1,keepdim=True) / torch.sum(torch.square(y_pred - y_true),dim=-1,keepdim=True) + 1e-8).mean()
        
        return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr




if __name__=='__main__':
    a = torch.randn(2,10000)
    b = torch.randn(2, 9990)
    loss_func = loss_sisnr()
    loss = loss_func(a,b)
    print(loss)
    
    S_ = torch.randn(3, 257, 91, 2)
    S = torch.randn(3, 257, 91, 2)
    loss_func = loss_hybrid()
    loss = loss_func(S_, S)
    print(loss)