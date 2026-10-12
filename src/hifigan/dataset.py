import os
import random
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from src.common.audio import AudioMelConversions, pad_for_mel

class MelDataset(Dataset):
    """
    Dataset to train and fine-tune HiFi-GAN neural vocoder.
    In standard pre-training mode, extracts Mel-spectrograms from raw audio.
    In fine-tuning mode, loads cached Tacotron2 generated Mel-spectrograms to close domain discrepancy.
    """
    def __init__(self, 
                 path_to_manifest, 
                 segment_size=8192, 
                 n_fft=1024, 
                 num_mels=80,
                 hop_size=256, 
                 window_size=1024, 
                 sampling_rate=22050,  
                 fmin=0, 
                 fmax=8000, 
                 fmax_loss=None,
                 max_audio_magnitude=0.95, 
                 min_db=-100.0, 
                 max_scaled_abs=4.0,
                 finetuning=False,
                 path_to_saved_mels=None):

        self.audio_files = list(pd.read_csv(path_to_manifest)["file_path"])

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = window_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.max_audio_magnitude = max_audio_magnitude
        self.finetuning = finetuning
        self.path_to_saved_mels = path_to_saved_mels

        if self.finetuning and self.path_to_saved_mels is None:
            raise ValueError("When finetuning, provide Teacher Forcing Generations as .npy files")

        common_args = dict(
            num_mels=num_mels, 
            sampling_rate=sampling_rate,
            n_fft=n_fft, 
            window_size=window_size, 
            hop_size=hop_size, 
            fmin=fmin, 
            center=False,
            min_db=min_db, 
            max_scaled_abs=max_scaled_abs
        )

        self.audio_mel_conv = AudioMelConversions(
            **common_args,
            fmax=fmax,
        )

        self.audio_mel_conv_loss = AudioMelConversions(
            **common_args,
            fmax=fmax_loss,
        )
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, sampling_rate = torchaudio.load(filename)
        
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val * self.max_audio_magnitude

        if sampling_rate != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=self.sampling_rate)

        if not self.finetuning:
            if audio.shape[1] >= self.segment_size:
                max_audio_start = audio.shape[1] - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start + self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape[1]), 'constant')

            audio_padded_for_mel = pad_for_mel(audio, self.n_fft, self.hop_size)
            mel = self.audio_mel_conv.audio2mel(audio_padded_for_mel, do_norm=True)
            mel_loss = self.audio_mel_conv_loss.audio2mel(audio_padded_for_mel, do_norm=True)

            return mel.squeeze(0), audio, mel_loss.squeeze(0)

        else:
            base_id = os.path.splitext(os.path.basename(filename))[0]
            path_to_mel = os.path.join(self.path_to_saved_mels, f"{base_id}.npy")
            mel = torch.from_numpy(np.load(path_to_mel)).float()

            frames_per_mel_segment = self.segment_size // self.hop_size
     
            if audio.shape[1] >= self.segment_size and mel.shape[1] > frames_per_mel_segment:
                max_mel_start = max(0, mel.shape[1] - frames_per_mel_segment - 1)
                mel_start = random.randint(0, max_mel_start)
                mel = mel[:, mel_start:mel_start + frames_per_mel_segment]
                audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_mel_segment) * self.hop_size]
            else:
                mel = torch.nn.functional.pad(mel, (0, max(0, frames_per_mel_segment - mel.shape[1])), 'constant')
                audio = torch.nn.functional.pad(audio, (0, max(0, self.segment_size - audio.shape[1])), 'constant')
            
            audio_padded_for_mel = pad_for_mel(audio, self.n_fft, self.hop_size)
            mel_loss = self.audio_mel_conv_loss.audio2mel(audio_padded_for_mel, do_norm=True)
            
            return mel, audio, mel_loss.squeeze(0)
