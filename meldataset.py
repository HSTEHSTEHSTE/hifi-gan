import csv
import math
import random
from pathlib import Path

import librosa
import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    # LibriSpeech source audio is FLAC, which scipy.io.wavfile cannot read.
    # Keeping sr=None ensures the waveform stays aligned with WavLM frames.
    data, sampling_rate = librosa.load(full_path, sr=None, mono=True)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    key = str(fmax) + '_' + str(y.device)
    if key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
            fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    stft_args = dict(
        input=y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window[str(y.device)], center=center, pad_mode='reflect',
        normalized=False, onesided=True)
    try:
        # PyTorch >= 1.8: complex STFT output is the supported interface.
        spec = torch.stft(return_complex=True, **stft_args).abs().clamp_min_(1e-9)
    except TypeError:
        # The original project pins PyTorch 1.4, where torch.stft returns a
        # real/imaginary pair and does not accept return_complex.
        spec = torch.sqrt(torch.stft(**stft_args).pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    training_files = _read_manifest(a.input_training_file)
    validation_files = _read_manifest(a.input_validation_file)
    return training_files, validation_files


def _read_manifest(path):
    required_columns = {'audio_path', 'feat_path'}
    with open(path, newline='', encoding='utf-8') as manifest:
        reader = csv.DictReader(manifest)
        rows = list(reader)
        missing = required_columns.difference(reader.fieldnames or ())
    if missing:
        raise ValueError(f'CSV manifest {path} is missing columns: {sorted(missing)}')
    return rows


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, audio_root_path=None, feature_root_path=None):
        self.audio_files = training_files
        if shuffle:
            self.audio_files = self.audio_files.copy()
            random.Random(1234).shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        if audio_root_path is None or feature_root_path is None:
            raise ValueError('audio_root_path and feature_root_path are required for WavLM training')
        self.audio_root_path = Path(audio_root_path)
        self.feature_root_path = Path(feature_root_path)

    def __getitem__(self, index):
        row = self.audio_files[index]
        audio_path = self.audio_root_path / row['audio_path']
        feature_path = self.feature_root_path / row['feat_path']
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(audio_path)
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.unsqueeze(0)

        features = torch.load(feature_path, map_location='cpu').float()
        if features.ndim == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError(f'Expected WavLM features shaped (frames, channels), got {tuple(features.shape)} at {feature_path}')

        if self.split:
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)
            usable_frames = min(features.size(0), audio.size(1) // self.hop_size)
            if usable_frames >= frames_per_seg:
                frame_start = random.randint(0, usable_frames - frames_per_seg)
                features = features[frame_start:frame_start + frames_per_seg]
                audio_start = frame_start * self.hop_size
                audio = audio[:, audio_start:audio_start + self.segment_size]
            else:
                features = F.pad(features, (0, 0, 0, frames_per_seg - features.size(0)))
                audio = audio[:, :self.segment_size]
                audio = F.pad(audio, (0, self.segment_size - audio.size(1)))
        else:
            # Keep full-utterance validation targets and generated audio on
            # the same 320-sample WavLM frame grid.
            usable_frames = min(features.size(0), audio.size(1) // self.hop_size)
            features = features[:usable_frames]
            audio = audio[:, :usable_frames * self.hop_size]

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (features, audio.squeeze(0), row['audio_path'], mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
