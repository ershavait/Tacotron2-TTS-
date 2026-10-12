import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common.tokenizer import Tokenizer
from src.common.audio import (
    AudioMelConversions,
    load_wav,
    amp_to_db,
    db_to_amp,
    normalize,
    denormalize,
    build_padding_mask,
)

class TTSDataset(Dataset):
    """
    Dataset loader for LJSpeech style text-to-speech datasets.
    Extracts text tokens and corresponding normalized Mel-spectrograms.
    """
    def __init__(self, 
                 path_to_metadata,
                 sample_rate=22050,
                 n_fft=1024, 
                 window_size=1024, 
                 hop_size=256, 
                 fmin=0,
                 fmax=8000, 
                 num_mels=80, 
                 center=False, 
                 normalized=False, 
                 min_db=-100.0, 
                 max_scaled_abs=4.0):
        
        self.metadata = pd.read_csv(path_to_metadata)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax 
        self.num_mels = num_mels
        self.center = center
        self.normalized = normalized
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        tokenizer = Tokenizer()
        self.transcript_lengths = [len(tokenizer.encode(t)) for t in self.metadata["normalized_transcript"]]

        self.audio_proc = AudioMelConversions(num_mels=self.num_mels, 
                                              sampling_rate=self.sample_rate, 
                                              n_fft=self.n_fft, 
                                              window_size=self.win_size, 
                                              hop_size=self.hop_size, 
                                              fmin=self.fmin, 
                                              fmax=self.fmax, 
                                              center=self.center,
                                              min_db=self.min_db, 
                                              max_scaled_abs=self.max_scaled_abs)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        path_to_audio = sample["file_path"]
        transcript = sample["normalized_transcript"]

        audio = load_wav(path_to_audio, sr=self.sample_rate)
        mel = self.audio_proc.audio2mel(audio, do_norm=True)

        return transcript, mel.squeeze(0), path_to_audio

def TTSCollator():
    """
    Collate function to dynamic pad batch texts and Mel-spectrograms,
    sorting by descending text length for PyTorch packed sequence compatibility.
    """
    tokenizer = Tokenizer()

    def _collate_fn(batch):
        texts = [tokenizer.encode(b[0]) for b in batch]
        mels = [b[1] for b in batch]
        paths = [b[2] if len(b) > 2 else "" for b in batch]
        
        input_lengths = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]
        sorted_paths = [paths[i] for i in sorted_idx]

        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

        max_target_len = max(output_lengths).item()
        num_mels = mels[0].shape[0]
        
        mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
        gate_padded = torch.zeros((len(mels), max_target_len))

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t-1:] = 1.0
        
        mel_padded = mel_padded.transpose(1, 2)

        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            build_padding_mask(input_lengths),
            build_padding_mask(output_lengths),
            sorted_paths,
        )

    return _collate_fn

class BatchSampler:
    """
    Custom batch sampler creating shuffled batches of similar length samples to reduce padding overhead.
    """
    def __init__(self, dataset, batch_size, drop_last=False):
        self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches()

    def _make_batches(self):
        indices = [i for i in self.sampler]

        if self.drop_last:
            total_size = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:total_size]

        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        random_indices = torch.randperm(len(batches))
        return [batches[i] for i in random_indices]
    
    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)
