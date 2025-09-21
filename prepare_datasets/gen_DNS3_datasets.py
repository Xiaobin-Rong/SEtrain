#!/usr/bin/env python3
"""Generate DNS-style noisy/clean pairs from clean, noise, and RIR lists."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic DNS-style mixtures.")
    parser.add_argument("--flag", default="train", help="Dataset split name used in file naming (default: train).")
    parser.add_argument("--num", type=int, default=50000, help="Maximum number of mixtures to generate (default: 50000).")
    parser.add_argument("--wav-len", type=float, default=10.0, help="Target audio length in seconds (default: 10).")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate (default: 16000).")
    parser.add_argument("--random-start", action="store_true", default=False, help="Randomly crop within source audio. Off by default to avoid empty reads when clips are short.")
    parser.add_argument("--snr-min", type=float, default=-5.0, help="Lower bound for SNR sampling (default: -5 dB).")
    parser.add_argument("--snr-max", type=float, default=15.0, help="Upper bound for SNR sampling (default: 15 dB).")
    parser.add_argument("--save-root", type=Path, default=Path("/home/gzhu/data1/cyh/audio/SEtrain/datasets/DNS3"), help="Output directory where train_noisy/train_clean etc. reside.")
    parser.add_argument("--csv-root", type=Path, default=Path(__file__).parent, help="Directory containing <flag>_clean_dir.csv etc.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed controlling reproducibility.")
    parser.add_argument("--with-replacement", action="store_true", help="Sample clean/noise/RIR with replacement so entries may repeat.")
    return parser.parse_args()


def add_pyreverb(clean_speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    return reverb_speech[: clean_speech.shape[0]]


def mk_mixture(s1: np.ndarray, s2: np.ndarray, s1_ref: np.ndarray, snr: float, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    amp = 0.5 * np.random.rand() + 0.01
    s1_ref = amp * s1_ref / (np.max(np.abs(s1)) + eps)
    s1 = amp * s1 / (np.max(np.abs(s1)) + eps)
    norm_sig1 = s1

    norm_sig2 = s2 * np.sqrt(np.sum(s1 ** 2) + eps) / np.sqrt(np.sum(s2 ** 2) + eps)
    alpha = 10 ** (-snr * 1.5 / 20)

    mix = norm_sig1 + alpha * norm_sig2

    M = max(np.max(np.abs(mix)), np.max(np.abs(norm_sig1)), np.max(np.abs(alpha * norm_sig2))) + eps
    if M > 1.0:
        mix = mix / M
        norm_sig1 = norm_sig1 / M
        norm_sig2 = norm_sig2 / M
        s1_ref = s1_ref / M

    return mix, s1_ref


def read_mono(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32)


def ensure_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.size == 0:
        raise ValueError("Encountered empty audio while ensuring length.")
    if signal.size >= target_len:
        return signal
    repeat = math.ceil(target_len / signal.size)
    tiled = np.tile(signal, repeat)
    return tiled


def select_segment(signal: np.ndarray, target_len: int, use_random_start: bool) -> np.ndarray:
    signal = ensure_length(signal, target_len)
    max_start = signal.size - target_len
    if use_random_start and max_start > 0:
        start = np.random.randint(0, max_start + 1)
    else:
        start = 0
    segment = signal[start: start + target_len]
    if segment.size < target_len:
        segment = np.pad(segment, (0, target_len - segment.size))
    return np.asarray(segment, dtype=np.float32)


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)

    flag = args.flag
    target_len = int(args.wav_len * args.fs)
    snr_range = (args.snr_min, args.snr_max)

    csv_root = args.csv_root.expanduser().resolve()
    save_root = args.save_root.expanduser().resolve()

    clean_csv = csv_root / f"{flag}_clean_dir.csv"
    noise_csv = csv_root / f"{flag}_noise_dir.csv"
    rir_csv = csv_root / f"{flag}_rir_dir.csv"

    clean_entries = pd.read_csv(clean_csv)["file_dir"].tolist()
    noise_entries = pd.read_csv(noise_csv)["file_dir"].tolist()
    rir_entries = pd.read_csv(rir_csv)["file_dir"].tolist()

    clean_paths = [Path(p) for p in clean_entries]
    noise_paths = [Path(p) for p in noise_entries]
    rir_paths = [Path(p) for p in rir_entries]

    if args.with_replacement:
        if not clean_paths or not noise_paths or not rir_paths:
            raise ValueError("Clean, noise, and RIR lists must all be non-empty when sampling with replacement.")
        num_tot = args.num
    else:
        num_tot = min(args.num, len(clean_paths), len(noise_paths), len(rir_paths))
        if num_tot < args.num:
            print(f"[Warning] Only {num_tot} triplets available; requested {args.num}.")
        clean_paths = clean_paths[:num_tot]
        noise_paths = noise_paths[:num_tot]
        rir_paths = rir_paths[:num_tot]

    nfill = len(str(num_tot))
    snr_list = np.random.uniform(snr_range[0], snr_range[1], size=num_tot)

    noisy_dir = save_root / f"{flag}_noisy"
    clean_dir = save_root / f"{flag}_clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    info_clean: list[str] = []
    info_noise: list[str] = []
    info_snr: list[float] = []

    for idx in tqdm(range(num_tot)):
        if args.with_replacement:
            clean_path = clean_paths[np.random.randint(len(clean_paths))]
            noise_path = noise_paths[np.random.randint(len(noise_paths))]
            rir_path = rir_paths[np.random.randint(len(rir_paths))]
        else:
            clean_path = clean_paths[idx]
            noise_path = noise_paths[idx]
            rir_path = rir_paths[idx]

        clean_full = read_mono(clean_path)
        noise_full = read_mono(noise_path)
        rir = read_mono(rir_path)

        clean = select_segment(clean_full, target_len, args.random_start)
        noise = select_segment(noise_full, target_len, args.random_start)

        max_index = int(np.argmax(np.abs(rir)))
        rir = rir[max_index:]
        rir_e = rir[: min(int(100 * args.fs / 1000), len(rir))]

        rev_clean = add_pyreverb(clean, rir)
        drb_clean = add_pyreverb(clean, rir_e)

        mixture, target = mk_mixture(rev_clean, noise, drb_clean, snr_list[idx], eps=1e-8)

        file_id = str(idx + 1).zfill(nfill) + ".wav"
        sf.write(noisy_dir / file_id, mixture, args.fs)
        sf.write(clean_dir / file_id, target, args.fs)

        info_clean.append(str(clean_path))
        info_noise.append(str(noise_path))
        info_snr.append(float(snr_list[idx]))

    info = pd.DataFrame({
        "file_name": [str(idx + 1).zfill(nfill) + ".wav" for idx in range(num_tot)],
        "clean": info_clean,
        "noise": info_noise,
        "snr": info_snr,
    })

    info.to_csv(save_root / f"{flag}_INFO.csv", index=None)


if __name__ == "__main__":
    main()
