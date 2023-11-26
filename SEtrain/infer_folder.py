"""
conduct evaluation on a folder of WAV files, with computing SISNR, PESQ, ESTOI, and DNSMOS.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import toml
import torch
from tqdm import tqdm
import soundfile as sf
from omegaconf import OmegaConf
from model import DPCRN


cfg_yaml = OmegaConf.load('config.yaml')
test_folder = '/data/ssd0/xiaobin.rong/Datasets/DNS3/blind_test_set/dns-challenge-3-final-evaluation/wideband_16kHz/noisy_clips_wb_16kHz/'
test_wavnames = list(filter(lambda x: x.endswith("wav"), os.listdir(test_folder)))

cfg_toml = toml.load(cfg_yaml.network.cfg_toml) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netout_folder = f'{cfg_yaml.path.exp_folder}'
os.makedirs(netout_folder, exist_ok=True)

### load model
model = DPCRN(**cfg_toml['network_config'])
model.to(device)
checkpoint = torch.load(cfg_yaml.network.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

for param in model.parameters():
    param.requires_grad = False

### compute SISNR, PESQ and ESTOI
with torch.no_grad():
    for name in tqdm(test_wavnames):
        noisy, fs = sf.read(os.path.join(test_folder, name), dtype="float32")
        noisy = torch.stft(torch.from_numpy(noisy), **cfg_toml['FFT'], window=torch.hann_window(cfg_toml['FFT']['win_length']).pow(0.5))
        noisy = noisy.to(device)
        
        estimate= model(noisy[None, ...])  # (B,F,T,2)

        enhanced = torch.istft(estimate[..., 0] + 1j*estimate[..., 1], **cfg_toml['FFT'], window=torch.hann_window(cfg_toml['FFT']['win_length']).pow(0.5).to(device))
        out = enhanced.cpu().detach().numpy().squeeze()

        sf.write(os.path.join(netout_folder, name[:-4]+'_enh.wav'), out, fs)

### compute DNSMOS
os.chdir('DNSMOS')
out_dir = os.path.join(netout_folder, 'dnsmos_enhanced_p808.csv')
os.system(f'python dnsmos_local_p808.py -t {netout_folder} -o {out_dir}')

