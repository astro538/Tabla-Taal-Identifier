"""
Dataset loading, preprocessing, augmentation, and feature extraction
for Tabla Taal & Bol classification.
"""

import os
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import Config


def load_file_paths(data_dir: str):
    """Load all audio file paths and their labels from directory structure.
    
    Expects structure: data_dir/class_name/*.wav
    
    Returns:
        file_paths: list of absolute paths
        labels: list of integer labels
        class_names: list of class name strings (sorted)
    """
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ])

    file_paths = []
    labels = []

    for idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls_name)
        wav_files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(".wav")
        ])
        for wav_file in wav_files:
            file_paths.append(os.path.join(cls_dir, wav_file))
            labels.append(idx)

    return file_paths, labels, class_names


def load_and_preprocess_audio(file_path: str, config: Config) -> np.ndarray:
    """Load a single audio file and preprocess it.
    
    Steps:
        1. Load and resample to target sample rate
        2. Convert to mono
        3. Trim silence
        4. Normalize amplitude
        5. Pad or truncate to fixed length
    """
    try:
        audio, sr = librosa.load(
            file_path,
            sr=config.audio.sample_rate,
            mono=True
        )
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        # Return silence as fallback
        audio = np.zeros(config.target_length, dtype=np.float32)
        return audio

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=config.audio.top_db)

    # Normalize amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Pad or truncate to fixed length
    target_len = config.target_length
    if len(audio) > target_len:
        # Random crop during training for variety, center crop otherwise
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")

    return audio.astype(np.float32)


def extract_mel_spectrogram(audio: np.ndarray, config: Config) -> np.ndarray:
    """Extract mel spectrogram from preprocessed audio.
    
    Returns:
        mel_db: normalized mel spectrogram in dB, shape (n_mels, target_frames)
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        n_mels=config.audio.n_mels,
        power=2.0
    )

    # Convert to dB scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate time dimension to target_frames
    target_frames = config.target_frames
    if mel_db.shape[1] > target_frames:
        mel_db = mel_db[:, :target_frames]
    elif mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant",
                        constant_values=mel_db.min())

    # Normalize: zero mean, unit variance per sample
    mean = mel_db.mean()
    std = mel_db.std()
    if std > 0:
        mel_db = (mel_db - mean) / (std + 1e-6)
    else:
        mel_db = mel_db - mean

    return mel_db.astype(np.float32)


def augment_waveform(audio: np.ndarray, config: Config) -> np.ndarray:
    """Apply random waveform-level augmentations.
    
    Augmentations:
        1. Additive Gaussian noise
        2. Random gain adjustment
        3. Circular time shift
    """
    aug_audio = audio.copy()

    # 1. Add Gaussian noise (70% chance)
    if random.random() < 0.7:
        noise_factor = random.uniform(*config.augment.noise_factor_range)
        noise = np.random.randn(len(aug_audio)).astype(np.float32)
        aug_audio = aug_audio + noise_factor * noise

    # 2. Random gain (60% chance)
    if random.random() < 0.6:
        gain_db = random.uniform(*config.augment.gain_db_range)
        gain = 10 ** (gain_db / 20)
        aug_audio = aug_audio * gain

    # 3. Random time shift (50% chance)
    if random.random() < 0.5:
        max_shift = int(len(aug_audio) * config.augment.shift_fraction)
        shift = random.randint(-max_shift, max_shift)
        aug_audio = np.roll(aug_audio, shift)

    # Re-normalize
    max_val = np.max(np.abs(aug_audio))
    if max_val > 0:
        aug_audio = aug_audio / max_val

    return aug_audio


def apply_spec_augment(mel: np.ndarray, config: Config) -> np.ndarray:
    """Apply SpecAugment: frequency and time masking on mel spectrogram.
    
    This is the key augmentation technique for audio classification.
    Masks random bands in frequency and time dimensions.
    """
    mel = mel.copy()
    n_mels, n_time = mel.shape
    fill_value = mel.mean()

    # Frequency masking
    for _ in range(config.augment.n_freq_masks):
        f = random.randint(1, min(config.augment.freq_mask_param, n_mels - 1))
        f0 = random.randint(0, n_mels - f)
        mel[f0:f0 + f, :] = fill_value

    # Time masking
    for _ in range(config.augment.n_time_masks):
        t = random.randint(1, min(config.augment.time_mask_param, n_time - 1))
        t0 = random.randint(0, n_time - t)
        mel[:, t0:t0 + t] = fill_value

    return mel


class TablaDataset(Dataset):
    """PyTorch Dataset for Tabla audio classification.
    
    Precomputes mel spectrograms at initialization for fast training.
    Applies SpecAugment on-the-fly during training.
    Optionally creates augmented copies of training data via waveform augmentation.
    """

    def __init__(self, file_paths, labels, config: Config,
                 augment: bool = False, create_augmented_copies: bool = False):
        """
        Args:
            file_paths: list of audio file paths
            labels: list of integer labels
            config: Config object
            augment: if True, apply SpecAugment on-the-fly
            create_augmented_copies: if True, create augmented waveform copies
                                      (only for training set)
        """
        self.config = config
        self.augment = augment
        self.mel_specs = []
        self.labels = []

        desc = "Loading & processing audio"
        for path, label in tqdm(zip(file_paths, labels), total=len(file_paths),
                                desc=desc, leave=False):
            # Load and preprocess
            audio = load_and_preprocess_audio(path, config)

            # Extract mel spectrogram for original
            mel = extract_mel_spectrogram(audio, config)
            self.mel_specs.append(mel)
            self.labels.append(label)

            # Create augmented copies for training diversity
            if create_augmented_copies:
                for _ in range(config.augment.n_augmented_copies):
                    aug_audio = augment_waveform(audio, config)
                    aug_mel = extract_mel_spectrogram(aug_audio, config)
                    self.mel_specs.append(aug_mel)
                    self.labels.append(label)

        self.mel_specs = np.array(self.mel_specs)
        self.labels = np.array(self.labels)

        print(f"  Dataset: {len(self.mel_specs)} samples, "
              f"shape: {self.mel_specs[0].shape}, "
              f"augment: {augment}, copies: {create_augmented_copies}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mel_specs[idx]

        # Apply SpecAugment on-the-fly during training
        if self.augment:
            mel = apply_spec_augment(mel, self.config)

        # Convert to tensor: (1, n_mels, time) - single channel like grayscale image
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]

        return mel_tensor, label
