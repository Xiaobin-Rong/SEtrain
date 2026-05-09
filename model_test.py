import os
import shutil
import torch
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
from models.gtcrn_end2end import GTCRN as Model

def main(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    
    noisy_folder = cfg_infer.test_dataset.noisy_dir
    test_output_folder = cfg_infer.network.output_folder
    os.makedirs(test_output_folder, exist_ok=True)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    model = Model(**cfg_network['network_config']).to(device)
    checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    noisy_wavs = sorted(list(filter(lambda x: x.endswith("wav"), os.listdir(noisy_folder))))

    inf_scp_list = []
    for wav_name in tqdm(noisy_wavs):
        noisy, fs = sf.read(os.path.join(noisy_folder, wav_name), dtype='float32')
        
        input = torch.FloatTensor(noisy).unsqueeze(0).to(device)
        with torch.inference_mode():
            output  = model(input)
        enhanced = output.cpu().detach().numpy().squeeze()
        
        uid = wav_name.split(".wav")[0]
        enh_path = os.path.join(test_output_folder, uid + f"_enh.wav")
        
        inf_scp_list.append([uid, enh_path])
        
        sf.write(enh_path, enhanced, fs)
    
    # Save paths into scp file for evaluation
    with open(os.path.join(test_output_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_test.yaml')
    parser.add_argument('-D', '--device', default='0', help='Index of the gpu device')

    args = parser.parse_args()
    main(args)
