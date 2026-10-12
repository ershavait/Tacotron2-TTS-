import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np

def load_wav(path_to_audio, sr=22050):
    """
    Load an audio waveform and resample to the target sample rate if needed.
    Returns 1D Tensor of audio samples.
    """
    audio, orig_sr = torchaudio.load(path_to_audio)
    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)
    return audio.squeeze(0)

def amp_to_db(x, min_db=-100.0):
    """
    Convert linear amplitude spectrogram to decibel scale clamped at min_db.
    """
    clip_val = 10.0 ** (min_db / 20.0)
    return 20.0 * torch.log10(torch.clamp(x, min=clip_val))

def db_to_amp(x):
    """
    Convert decibels back to linear amplitude.
    """
    return 10.0 ** (x / 20.0)

def normalize(x, min_db=-100.0, max_abs_val=4.0):
    """
    Scale decibel spectrogram values between [-max_abs_val, max_abs_val].
    """
    x = (x - min_db) / -min_db
    x = 2.0 * max_abs_val * x - max_abs_val
    return torch.clip(x, min=-max_abs_val, max=max_abs_val)

def denormalize(x, min_db=-100.0, max_abs_val=4.0):
    """
    Reverse normalization back to decibel scale.
    """
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / (2.0 * max_abs_val)
    return x * -min_db + min_db

def pad_for_mel(x, n_fft, hop_size):
    """
    Reflective padding for audio waveform to align STFT center frames.
    """
    pad = (n_fft - hop_size) // 2
    return F.pad(x, (pad, pad), mode="reflect")

def build_padding_mask(lengths):
    """
    Create boolean mask where True indicates padded timesteps.
    """
    B = lengths.size(0)
    T = torch.max(lengths).item()
    mask = torch.zeros(B, T, dtype=torch.bool, device=lengths.device)
    for i in range(B):
        mask[i, lengths[i]:] = True
    return mask

class AudioMelConversions:
    """
    Bidirectional conversion between raw audio waveforms and Mel-spectrograms.
    Supports Griffin-Lim iterative audio synthesis for quick baseline evaluation.
    """
    def __init__(self,
                 num_mels=80,
                 sampling_rate=22050,
                 n_fft=1024,
                 window_size=1024,
                 hop_size=256,
                 fmin=0,
                 fmax=8000,
                 center=False,
                 min_db=-100.0,
                 max_scaled_abs=4.0):
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

    def _get_spec2mel_proj(self):
        mel = librosa.filters.mel(sr=self.sampling_rate,
                                  n_fft=self.n_fft,
                                  n_mels=self.num_mels,
                                  fmin=self.fmin,
                                  fmax=self.fmax)
        return torch.from_numpy(mel).float()

    def audio2mel(self, audio, do_norm=False):
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(input=audio,
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_size,
                                 win_length=self.window_size,
                                 window=torch.hann_window(self.window_size).to(audio.device),
                                 center=self.center,
                                 pad_mode="reflect",
                                 normalized=False,
                                 onesided=True,
                                 return_complex=True)

        spectrogram = torch.abs(spectrogram)
        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)
        mel = amp_to_db(mel, self.min_db)

        if do_norm:
            mel = normalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        return mel

    def mel2audio(self, mel, do_denorm=False, griffin_lim_iters=60):
        if do_denorm:
            mel = denormalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        mel = db_to_amp(mel)
        spectrogram = torch.matmul(self.mel2spec.to(mel.device), mel).cpu().numpy()

        audio = librosa.griffinlim(S=spectrogram,
                                   n_iter=griffin_lim_iters,
                                   hop_length=self.hop_size,
                                   win_length=self.window_size,
                                   n_fft=self.n_fft,
                                   window="hann")

        audio *= 32767.0 / max(0.01, np.max(np.abs(audio)))
        return audio.astype(np.int16)
