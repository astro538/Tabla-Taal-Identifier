"""
Configuration for Tabla Taal & Bol Identification.
All hyperparameters and settings in one place.
"""

from dataclasses import dataclass, field
import os


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    top_db: float = 25.0  # for silence trimming


@dataclass
class AugmentConfig:
    """Data augmentation configuration."""
    # SpecAugment parameters
    n_freq_masks: int = 2
    freq_mask_param: int = 20      # max frequency bands to mask
    n_time_masks: int = 2
    time_mask_param: int = 30      # max time steps to mask

    # Waveform augmentation (applied during preprocessing)
    n_augmented_copies: int = 5    # number of augmented copies per original
    noise_factor_range: tuple = (0.002, 0.008)
    gain_db_range: tuple = (-6, 6)
    shift_fraction: float = 0.15   # max shift as fraction of audio length


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12             # early stopping patience
    n_folds: int = 5               # k-fold cross-validation
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.3       # mixup augmentation strength
    gradient_clip: float = 1.0
    scheduler: str = "cosine"      # 'cosine' or 'step'
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class Config:
    """Main configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Task configuration
    task: str = "bol"              # 'bol' or 'taal'
    num_classes: int = 8

    # Duration settings (seconds)
    bol_max_duration: float = 3.0
    taal_max_duration: float = 8.0

    # Paths
    dataset_root: str = ""
    output_dir: str = "results"
    save_models: bool = True

    # Models to train
    model_names: list = field(default_factory=lambda: [
        "ResidualCNN", "CRNN", "AttentionCRNN", "ASTLite"
    ])

    # Random seed
    seed: int = 42

    @property
    def max_duration(self) -> float:
        return self.bol_max_duration if self.task == "bol" else self.taal_max_duration

    @property
    def target_length(self) -> int:
        """Target audio length in samples."""
        return int(self.max_duration * self.audio.sample_rate)

    @property
    def target_frames(self) -> int:
        """Target number of mel spectrogram frames."""
        return 1 + self.target_length // self.audio.hop_length

    @property
    def task_data_dir(self) -> str:
        subdir = "Bols" if self.task == "bol" else "Taals"
        return os.path.join(self.dataset_root, subdir)

    @property
    def task_output_dir(self) -> str:
        return os.path.join(self.output_dir, self.task)

    def __post_init__(self):
        if not self.dataset_root:
            # Auto-detect dataset path
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.dataset_root = os.path.join(
                base, "Tabla-Taal-and-Bol-Identification", "Dataset"
            )
