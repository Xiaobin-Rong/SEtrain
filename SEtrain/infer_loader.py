import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

import toml
import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from score_utils import sisnr
from omegaconf import OmegaConf
from datasets import MyDataset
from model import DPCRN


@torch.no_grad()
def infer(cfg_yaml):

    save_wavs = input('>>> Save wavs? (y/n) ')
    if save_wavs == 'y':
        mark = input('>>> Please enter a tag for the saved wav names: ')

    cfg_toml = toml.load(cfg_yaml.network.cfg_toml)
    cfg_toml['validation_dataset']['train_folder'] = '/data/ssd0/xiaobin.rong/Datasets/DNS3/test/'
    cfg_toml['validation_dataset']['num_tot'] = 0         # all utterances
    cfg_toml['validation_dataset']['wav_len'] = 0         # full wav length
    cfg_toml['validation_dataloader']['batch_size'] = 1   # one utterence once

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netout_folder = f'{cfg_yaml.path.exp_folder}'
    os.system(f'rm {netout_folder}/*.wav')
    os.makedirs(netout_folder, exist_ok=True)

    validation_dataset = MyDataset(**cfg_toml['validation_dataset'])
    validation_filename = validation_dataset.file_name
    
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, **cfg_toml['validation_dataloader'])

    ### load model
    model = DPCRN(**cfg_toml['network_config'])
    model.to(device)
    checkpoint = torch.load(cfg_yaml.network.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    ### compute SISNR, PESQ, and ESTOI
    INFO1 = []
    INFO = pd.read_csv(os.path.join(cfg_toml['validation_dataset']['train_folder'], 'INFO.csv'))
    for step, (mixture, target) in enumerate(tqdm(validation_dataloader)):
            
        mixture = mixture.to(device)
        target = target.to(device)
        
        estimate= model(mixture)                       # [B, F, T, 2]

        enhanced = torch.istft(estimate[..., 0] + 1j*estimate[..., 1], **cfg_toml['FFT'], window=torch.hann_window(cfg_toml['FFT']['win_length']).pow(0.5).to(device))
        clean = torch.istft(target[..., 0] + 1j*target[..., 1], **cfg_toml['FFT'], window=torch.hann_window(cfg_toml['FFT']['win_length']).pow(0.5).to(device))

        out = enhanced.cpu().detach().numpy().squeeze()
        clean = clean.cpu().detach().numpy().squeeze()

        # out = torch.clamp(out, -1, 1)
        # out = out / out.max() * 0.5

        sisnr_score = sisnr(out, clean)
        pesq_score = pesq(16000, clean, out, 'wb')
        estoi_score = stoi(clean, out, 16000, extended=True)

        ## save wavs
        if save_wavs == 'y':
            save_name = "{}_{}_{:.2f}_{:.2f}_{:.2f}.wav".format(validation_filename[step], mark, sisnr_score, pesq_score, estoi_score)
        
            sf.write(
                os.path.join(netout_folder, save_name), out, cfg_toml['listener']['listener_sr'])
        
        ## save infos
        file_name = validation_filename[step]
        INFO1.append([file_name, sisnr_score, pesq_score,  estoi_score])
    
    INFO1 = pd.DataFrame(INFO1, columns=['file_name', 'sisnr', 'pesq', 'estoi'])
    INFO2 = pd.merge(INFO, INFO1)
    INFO2.to_csv(os.path.join(netout_folder, 'INFO2.csv'), index=None)

    ### compute DNSMOS
    os.chdir('DNSMOS')
    out_dir = os.path.join(netout_folder, 'dnsmos_enhanced_p808.csv')
    os.system(f'python dnsmos_local_p808.py -t {netout_folder} -o {out_dir}')
    
    
if __name__ == "__main__":
    cfg_yaml = OmegaConf.load('config.yaml')
    infer(cfg_yaml)
    
