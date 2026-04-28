import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from tqdm import tqdm

def add_pyreverb(clean_speech, rir):
    # max_index = np.argmax(np.abs(rir))
    # rir = rir[max_index:]
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[: clean_speech.shape[0]]

    return reverb_speech


def mk_mixture(s1, s2, s1_ref, snr, eps=1e-8):
    """
    s1: reverbrant speech
    s2: reverbrant speech or noise
    s1_ref: s1 with low reverbration, as target in training
    """
    amp = 0.5 * np.random.rand() + 0.01
    s1_ref = amp * s1_ref / (np.max(np.abs(s1)) + eps) 
    s1 = amp * s1 / (np.max(np.abs(s1)) + eps) 
    norm_sig1 = s1

    norm_sig2 = s2 * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s2 ** 2) + eps)
    alpha = 10**(-snr*1.5 / 20)
    # freq_num = np.random.randint(0, 4)
    # sins = np.zeros(len(s1))
    # if freq_num > 1:
    #     freq = np.random.choice(range(50, 8000), freq_num)
    #     for f in freq:
    #         s_sin = np.sin(2*np.pi * f * np.arange(len(s1)) / 16000)
    #         sins = sins + (0.5*np.random.rand() + 0.5) * alpha * s_sin * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s_sin ** 2) + eps)
    
    # mix = norm_sig1 + alpha * norm_sig2 + sins
    mix = norm_sig1 + alpha * norm_sig2
    
    M = max(np.max(abs(mix)), np.max(abs(norm_sig1)), np.max(abs(alpha*norm_sig2))) + eps
    if M > 1.0:    
        mix = mix / M
        norm_sig1 = norm_sig1 / M
        norm_sig2 = norm_sig2 / M
        s1_ref = s1_ref / M

    return mix, s1_ref


if __name__ == "__main__":
    np.random.seed(10)
    
    flag = 'train'
    num_tot = 50000
    nfill = len(str(num_tot))
    
    fs = 16000
    wav_len = 10  # in seconds
    random_start = True
    snr_range = [-5, 15]
    
    save_root = '/data/ssd0/xiaobin.rong/Datasets/DNS3/'
    
    data_root = './'
    clean_csv = os.path.join(data_root, f'{flag}_clean_dir.csv')
    noise_csv = os.path.join(data_root, f'{flag}_noise_dir.csv')
    rir_csv = os.path.join(data_root, f'{flag}_rir_dir.csv')

    clean_list = pd.read_csv(clean_csv)['file_dir'].tolist()[: num_tot]
    noise_list = pd.read_csv(noise_csv)['file_dir'].tolist()[: num_tot]
    rir_list = pd.read_csv(rir_csv)['file_dir'].tolist()[: num_tot]
    snr_list = np.random.uniform(snr_range[0], snr_range[1], size=num_tot)
    
    info = pd.DataFrame([str(idx+1).zfill(nfill)+'.wav' for idx in range(num_tot)], columns=['file_name'])
    info['clean'] = clean_list
    info['noise'] = noise_list
    info['snr'] = snr_list
    
    info.to_csv(os.path.join(save_root, f'{flag}_INFO.csv'), index=None)
    
    for idx in tqdm(range(num_tot)):
        
        if random_start:
            start_s = int(np.random.uniform(0, 15 - wav_len)) * fs
            start_n = int(np.random.uniform(0, 30 - wav_len)) * fs
        else:
            start_s = 0
            start_n = 0
        
        clean = sf.read(clean_list[idx], dtype='float32', start=start_s, stop=start_s + wav_len*fs)[0]
        noise = sf.read(noise_list[idx], dtype='float32', start=start_n, stop=start_n + wav_len*fs)[0]
        rir = sf.read(rir_list[idx], dtype='float32')[0]
        
        if len(rir.shape)>1:
            rir = rir[:, 0]
        max_index = np.argmax(np.abs(rir)) # find index where rir is max, response from here
        rir = rir[max_index:] 
        rir_e = rir[:min(int(100 * 16000 / 1000), len(rir))]  # rir_e: early rir, 选取前100ms的rir，用来生成低混响的干净语音 
        
        rev_clean = add_pyreverb(clean, rir)   # reverbrant clean speech
        drb_clean = add_pyreverb(clean, rir_e) # clean speech with low reverbration
        
        mixture, target = mk_mixture(rev_clean, noise, drb_clean, snr_list[idx], eps=1e-8)
        
        sf.write(os.path.join(save_root, f'{flag}_noisy', str(idx+1).zfill(nfill)+'.wav'), mixture, fs)
        sf.write(os.path.join(save_root, f'{flag}_clean', str(idx+1).zfill(nfill)+'.wav'), target, fs)
    
    