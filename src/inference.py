"""
Production Inference Engine for Tabla Taal & Bol Classification.

Loads the best trained models and provides a clean API for predictions
from audio files or raw bytes. Designed for integration with FastAPI
backend and future mobile app deployment.
"""

import os
import glob
import time
import io
import numpy as np
import librosa
import torch
import torch.nn.functional as F

from src.models import get_model


class TablaPredictor:
    """Production-grade inference engine for Tabla classification.
    
    Loads the best-performing model fold for both Bol and Taal tasks,
    and provides methods to predict from file paths or raw audio bytes.
    
    Usage:
        predictor = TablaPredictor(models_dir="results")
        result = predictor.predict_from_file("tabla_recording.wav", task="both")
        print(result)
    """

    BOL_CLASSES = ['dha', 'dhin', 'ga', 'ka', 't', 'ta', 'tin', 'tit']
    TAAL_CLASSES = [
        'addhatrital', 'bhajani', 'dadra', 'deepchandi',
        'ektal', 'jhaptal', 'rupak', 'trital'
    ]

    # Display names for cleaner output
    BOL_DISPLAY = {
        'dha': 'Dha', 'dhin': 'Dhin', 'ga': 'Ga', 'ka': 'Ka',
        't': 'T', 'ta': 'Ta', 'tin': 'Tin', 'tit': 'Tit'
    }
    TAAL_DISPLAY = {
        'addhatrital': 'Addha Trital', 'bhajani': 'Bhajani',
        'dadra': 'Dadra', 'deepchandi': 'Deepchandi',
        'ektal': 'Ektal', 'jhaptal': 'Jhaptal',
        'rupak': 'Rupak', 'trital': 'Trital'
    }

    # Taal beat structures for display
    TAAL_INFO = {
        'addhatrital': {'beats': 14, 'vibhag': '2-4-4-4'},
        'bhajani': {'beats': 8, 'vibhag': '4-4'},
        'dadra': {'beats': 6, 'vibhag': '3-3'},
        'deepchandi': {'beats': 14, 'vibhag': '3-4-3-4'},
        'ektal': {'beats': 12, 'vibhag': '4-4-4'},
        'jhaptal': {'beats': 10, 'vibhag': '2-3-2-3'},
        'rupak': {'beats': 7, 'vibhag': '3-2-2'},
        'trital': {'beats': 16, 'vibhag': '4-4-4-4'},
    }

    # Standard Theka (Bol patterns) for each Taal
    # These are the canonical Bol sequences that define each Taal cycle
    TAAL_BOL_PATTERNS = {
        'dadra': {
            'theka': 'Dha Dhin Na | Dha Tin Na',
            'bols': ['Dha', 'Dhin', 'Na', 'Dha', 'Tin', 'Na'],
            'vibhag_markers': [0, 3],  # beat indices where vibhag starts
        },
        'trital': {
            'theka': 'Dha Dhin Dhin Dha | Dha Dhin Dhin Dha | Dha Tin Tin Ta | Ta Dhin Dhin Dha',
            'bols': ['Dha', 'Dhin', 'Dhin', 'Dha', 'Dha', 'Dhin', 'Dhin', 'Dha',
                     'Dha', 'Tin', 'Tin', 'Ta', 'Ta', 'Dhin', 'Dhin', 'Dha'],
            'vibhag_markers': [0, 4, 8, 12],
        },
        'rupak': {
            'theka': 'Tin Tin Na | Dhin Na | Dhin Na',
            'bols': ['Tin', 'Tin', 'Na', 'Dhin', 'Na', 'Dhin', 'Na'],
            'vibhag_markers': [0, 3, 5],
        },
        'ektal': {
            'theka': 'Dhin Dhin Dha Dha | Tin Tin Na Na | Dhin Dhin Dha Dha',
            'bols': ['Dhin', 'Dhin', 'Dha', 'Dha', 'Tin', 'Tin',
                     'Na', 'Na', 'Dhin', 'Dhin', 'Dha', 'Dha'],
            'vibhag_markers': [0, 4, 8],
        },
        'jhaptal': {
            'theka': 'Dhi Na | Dhi Dhi Na | Ti Na | Dhi Dhi Na',
            'bols': ['Dhi', 'Na', 'Dhi', 'Dhi', 'Na', 'Ti', 'Na', 'Dhi', 'Dhi', 'Na'],
            'vibhag_markers': [0, 2, 5, 7],
        },
        'bhajani': {
            'theka': 'Dha Ge Na Ti | Na Ka Dhi Na',
            'bols': ['Dha', 'Ge', 'Na', 'Ti', 'Na', 'Ka', 'Dhi', 'Na'],
            'vibhag_markers': [0, 4],
        },
        'deepchandi': {
            'theka': 'Dha Dhin - Dha | Dha Dhin - Ta | Tin - Dha Dha | Dhin -',
            'bols': ['Dha', 'Dhin', '-', 'Dha', 'Dha', 'Dhin', '-',
                     'Ta', 'Tin', '-', 'Dha', 'Dha', 'Dhin', '-'],
            'vibhag_markers': [0, 3, 7, 10],
        },
        'addhatrital': {
            'theka': 'Dha Dhin | Dha Dha Tin Ta | Ta Tin Dha Dha | Dhin Dha Dha Tin',
            'bols': ['Dha', 'Dhin', 'Dha', 'Dha', 'Tin', 'Ta',
                     'Ta', 'Tin', 'Dha', 'Dha', 'Dhin', 'Dha', 'Dha', 'Tin'],
            'vibhag_markers': [0, 2, 6, 10],
        },
    }

    def __init__(self, models_dir="results", device=None):
        """Initialize predictor by loading best models.
        
        Args:
            models_dir: root directory containing bol/ and taal/ result folders
            device: torch device (auto-detected if None)
        """
        self.device = device or self._detect_device()

        # Audio processing config (must match training config)
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.top_db = 25.0

        # Task-specific durations
        self.bol_max_duration = 3.0     # seconds
        self.taal_max_duration = 8.0    # seconds

        # Load models
        print("Loading Tabla classification models...")
        self.bol_model, self.bol_model_name, self.bol_model_acc = self._load_best_model(
            os.path.join(models_dir, "bol", "models"),
            "ResidualCNN", num_classes=8
        )
        self.taal_model, self.taal_model_name, self.taal_model_acc = self._load_best_model(
            os.path.join(models_dir, "taal", "models"),
            "CRNN", num_classes=8
        )
        print(f"  ✓ Bol model:  {self.bol_model_name} (val_acc: {self.bol_model_acc:.2%})")
        print(f"  ✓ Taal model: {self.taal_model_name} (val_acc: {self.taal_model_acc:.2%})")
        print(f"  ✓ Device: {self.device}")
        print("Models loaded successfully!")

    @staticmethod
    def _detect_device():
        """Auto-detect the best available compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_best_model(self, models_dir, model_name, num_classes):
        """Load the best fold of a given model architecture.
        
        Scans all fold checkpoints and picks the one with highest val accuracy.
        """
        fold_files = sorted(glob.glob(
            os.path.join(models_dir, f"{model_name}_fold*.pth")
        ))

        if not fold_files:
            raise FileNotFoundError(
                f"No model files found for {model_name} in {models_dir}. "
                f"Please run training first with: python train.py"
            )

        best_acc = 0.0
        best_state = None
        best_fold = -1

        for fold_file in fold_files:
            checkpoint = torch.load(
                fold_file, map_location=self.device, weights_only=False
            )
            acc = checkpoint.get("best_val_acc", 0.0)
            if acc > best_acc:
                best_acc = acc
                best_state = checkpoint["model_state_dict"]
                best_fold = int(fold_file.split("fold")[-1].split(".")[0])

        # Create model and load weights
        model = get_model(model_name, num_classes=num_classes)
        model.load_state_dict(best_state)
        model.to(self.device)
        model.eval()

        name = f"{model_name} (fold {best_fold})"
        return model, name, best_acc

    def _load_audio(self, source):
        """Load audio from file path or bytes.
        
        Handles WAV, MP3, OGG, FLAC and other formats via librosa/soundfile.
        
        Returns:
            numpy array of audio samples at self.sample_rate
        """
        if isinstance(source, (str, os.PathLike)):
            audio, _ = librosa.load(source, sr=self.sample_rate, mono=True)
        elif isinstance(source, bytes):
            audio, _ = librosa.load(
                io.BytesIO(source), sr=self.sample_rate, mono=True
            )
        elif isinstance(source, io.BytesIO):
            audio, _ = librosa.load(source, sr=self.sample_rate, mono=True)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        return audio

    def _preprocess_audio(self, audio, max_duration):
        """Preprocess audio: trim, normalize, pad/truncate.
        
        Must match the preprocessing done during training.
        """
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=self.top_db)

        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Pad or truncate to fixed length
        target_len = int(max_duration * self.sample_rate)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")

        return audio.astype(np.float32)

    def _extract_mel_spectrogram(self, audio, max_duration):
        """Extract normalized mel spectrogram.
        
        Must match the feature extraction done during training.
        """
        target_frames = 1 + int(max_duration * self.sample_rate) // self.hop_length

        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad/truncate time dimension
        if mel_db.shape[1] > target_frames:
            mel_db = mel_db[:, :target_frames]
        elif mel_db.shape[1] < target_frames:
            pad_width = target_frames - mel_db.shape[1]
            mel_db = np.pad(
                mel_db, ((0, 0), (0, pad_width)),
                mode="constant", constant_values=mel_db.min()
            )

        # Normalize to zero mean, unit variance
        mean = mel_db.mean()
        std = mel_db.std()
        if std > 0:
            mel_db = (mel_db - mean) / (std + 1e-6)
        else:
            mel_db = mel_db - mean

        return mel_db.astype(np.float32)

    def _classify(self, audio, task):
        """Run classification on preprocessed audio for a given task.
        
        Returns dict with label, confidence, and top-k predictions.
        """
        if task == "bol":
            model = self.bol_model
            max_dur = self.bol_max_duration
            classes = self.BOL_CLASSES
            display = self.BOL_DISPLAY
        else:
            model = self.taal_model
            max_dur = self.taal_max_duration
            classes = self.TAAL_CLASSES
            display = self.TAAL_DISPLAY

        # Preprocess and extract features
        processed = self._preprocess_audio(audio, max_dur)
        mel = self._extract_mel_spectrogram(processed, max_dur)

        # Create tensor: (1, 1, n_mels, time)
        x = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Build results
        sorted_idx = np.argsort(probs)[::-1]
        top_k = []
        for i in sorted_idx:
            entry = {
                "label": classes[i],
                "display_name": display.get(classes[i], classes[i]),
                "confidence": round(float(probs[i]), 4)
            }
            # Add taal info if applicable
            if task == "taal" and classes[i] in self.TAAL_INFO:
                entry["beats"] = self.TAAL_INFO[classes[i]]["beats"]
                entry["vibhag"] = self.TAAL_INFO[classes[i]]["vibhag"]
            top_k.append(entry)

        return {
            "label": classes[sorted_idx[0]],
            "display_name": display.get(classes[sorted_idx[0]], classes[sorted_idx[0]]),
            "confidence": round(float(probs[sorted_idx[0]]), 4),
            "top_k": top_k
        }

    def predict_from_file(self, file_path, task="both"):
        """Predict Taal/Bol from an audio file.
        
        Args:
            file_path: path to audio file (WAV, MP3, OGG, FLAC, etc.)
            task: "bol", "taal", or "both"
        
        Returns:
            dict with prediction results and metadata
        """
        start_time = time.time()

        # Load audio
        audio = self._load_audio(file_path)
        duration = len(audio) / self.sample_rate

        # Run predictions
        results = {"status": "success", "task": task}

        if task in ("bol", "both"):
            results["bol"] = self._classify(audio, "bol")

        if task in ("taal", "both"):
            results["taal"] = self._classify(audio, "taal")

        # Metadata
        results["audio_info"] = {
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
            "source": os.path.basename(file_path) if isinstance(file_path, str) else "uploaded"
        }
        results["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)

        return results

    def predict_from_bytes(self, audio_bytes, task="both"):
        """Predict Taal/Bol from raw audio bytes.
        
        Args:
            audio_bytes: raw audio file bytes
            task: "bol", "taal", or "both"
        
        Returns:
            dict with prediction results and metadata
        """
        start_time = time.time()

        # Load audio
        audio = self._load_audio(audio_bytes)
        duration = len(audio) / self.sample_rate

        # Run predictions
        results = {"status": "success", "task": task}

        if task in ("bol", "both"):
            results["bol"] = self._classify(audio, "bol")

        if task in ("taal", "both"):
            results["taal"] = self._classify(audio, "taal")

        # Metadata
        results["audio_info"] = {
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
            "source": "uploaded"
        }
        results["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)

        return results

    # =========================================================================
    # Sequence Prediction (frame-by-frame Bol detection)
    # =========================================================================

    def _detect_onsets(self, audio, min_gap_ms=80):
        """Detect onset positions (individual Tabla strokes) in audio.
        
        Uses librosa's onset detection with backtracking to find the precise
        start of each percussive hit.
        
        Args:
            audio: mono audio at self.sample_rate
            min_gap_ms: minimum gap between onsets in milliseconds
        
        Returns:
            numpy array of onset times in seconds
        """
        # Onset detection with backtracking for precise timing
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=self.sample_rate,
            units='frames',
            hop_length=self.hop_length,
            backtrack=True,
            pre_max=3, post_max=3,
            pre_avg=3, post_avg=5,
            delta=0.05, wait=int(min_gap_ms * self.sample_rate / (1000 * self.hop_length))
        )

        onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Filter onsets that are too close together
        if len(onset_times) > 1:
            min_gap = min_gap_ms / 1000.0
            filtered = [onset_times[0]]
            for t in onset_times[1:]:
                if t - filtered[-1] >= min_gap:
                    filtered.append(t)
            onset_times = np.array(filtered)

        return onset_times

    def _classify_segment(self, audio_segment):
        """Classify a single audio segment as a Bol.
        
        Args:
            audio_segment: raw audio for one stroke
        
        Returns:
            dict with label, display_name, confidence, top_3
        """
        # Normalize
        max_val = np.max(np.abs(audio_segment))
        if max_val > 0:
            audio_segment = audio_segment / max_val

        # Pad/truncate to bol_max_duration
        target_len = int(self.bol_max_duration * self.sample_rate)
        if len(audio_segment) > target_len:
            audio_segment = audio_segment[:target_len]
        elif len(audio_segment) < target_len:
            audio_segment = np.pad(
                audio_segment, (0, target_len - len(audio_segment)), mode="constant"
            )

        audio_segment = audio_segment.astype(np.float32)

        # Extract mel spectrogram
        mel = self._extract_mel_spectrogram(audio_segment, self.bol_max_duration)

        # Model inference
        x = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.bol_model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        sorted_idx = np.argsort(probs)[::-1]

        # Top 3 for compact response
        top_3 = []
        for i in sorted_idx[:3]:
            top_3.append({
                "label": self.BOL_CLASSES[i],
                "display_name": self.BOL_DISPLAY.get(self.BOL_CLASSES[i], self.BOL_CLASSES[i]),
                "confidence": round(float(probs[i]), 4)
            })

        return {
            "label": self.BOL_CLASSES[sorted_idx[0]],
            "display_name": self.BOL_DISPLAY.get(
                self.BOL_CLASSES[sorted_idx[0]], self.BOL_CLASSES[sorted_idx[0]]
            ),
            "confidence": round(float(probs[sorted_idx[0]]), 4),
            "top_3": top_3
        }

    def predict_sequence(self, audio, segment_duration=0.5):
        """Predict Bol sequence from audio using onset detection.
        
        Segments the audio into individual Tabla strokes using onset detection,
        classifies each segment, and returns a timestamped timeline.
        
        Args:
            audio: mono audio array at self.sample_rate
            segment_duration: duration in seconds to extract per onset
        
        Returns:
            list of {time, end_time, bol, display_name, confidence, top_3}
        """
        # Detect onsets
        onset_times = self._detect_onsets(audio)
        duration = len(audio) / self.sample_rate

        if len(onset_times) == 0:
            return []

        timeline = []

        for i, onset in enumerate(onset_times):
            # Determine segment boundaries
            start_sample = int(onset * self.sample_rate)

            # End is either next onset or onset + segment_duration
            if i + 1 < len(onset_times):
                end_time = min(onset + segment_duration, onset_times[i + 1])
            else:
                end_time = min(onset + segment_duration, duration)

            end_sample = int(end_time * self.sample_rate)

            # Extract segment
            segment = audio[start_sample:end_sample]

            # Skip very short segments (likely noise)
            if len(segment) < int(0.03 * self.sample_rate):
                continue

            # Classify
            result = self._classify_segment(segment)

            timeline.append({
                "time": round(float(onset), 3),
                "end_time": round(float(end_time), 3),
                "bol": result["label"],
                "display_name": result["display_name"],
                "confidence": result["confidence"],
                "top_3": result["top_3"]
            })

        return timeline

    def predict_sequence_from_file(self, file_path, include_taal=True):
        """Full sequence prediction from file: Taal + Bol timeline.
        
        Args:
            file_path: path to audio file
            include_taal: whether to also predict the overall Taal
        
        Returns:
            dict with taal prediction, bol_sequence timeline, and metadata
        """
        start_time = time.time()

        audio = self._load_audio(file_path)
        duration = len(audio) / self.sample_rate

        results = {"status": "success"}

        # Overall Taal classification
        if include_taal:
            results["taal"] = self._classify(audio, "taal")

        # Bol sequence timeline
        results["bol_sequence"] = self.predict_sequence(audio)
        results["bol_count"] = len(results["bol_sequence"])

        # Metadata
        results["audio_info"] = {
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
            "source": os.path.basename(file_path) if isinstance(file_path, str) else "uploaded"
        }
        results["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)

        return results

    def predict_sequence_from_bytes(self, audio_bytes, include_taal=True):
        """Full sequence prediction from bytes."""
        start_time = time.time()

        audio = self._load_audio(audio_bytes)
        duration = len(audio) / self.sample_rate

        results = {"status": "success"}

        if include_taal:
            results["taal"] = self._classify(audio, "taal")

        results["bol_sequence"] = self.predict_sequence(audio)
        results["bol_count"] = len(results["bol_sequence"])

        results["audio_info"] = {
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
            "source": "uploaded"
        }
        results["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)

        return results

    # =========================================================================
    # Combined Prediction (Template + Model)
    # =========================================================================

    def _get_played_bols(self, taal_label):
        """Get the played Bols for a Taal (excluding rests '-')."""
        pattern = self.TAAL_BOL_PATTERNS.get(taal_label)
        if not pattern:
            return None
        return [b for b in pattern['bols'] if b != '-']

    def _align_onsets_to_template(self, onset_count, taal_label):
        """Align detected onset count to template Bol pattern.
        
        Returns list of Bol names for each onset, handling:
        - Exact match: 1:1 mapping
        - Multiple cycles: cyclic wrapping
        - Partial/extra onsets: best-effort proportional mapping
        """
        played_bols = self._get_played_bols(taal_label)
        if not played_bols:
            return [None] * onset_count

        cycle_len = len(played_bols)

        if onset_count == 0:
            return []

        # Determine number of cycles
        n_cycles = max(1, round(onset_count / cycle_len))
        expected_total = n_cycles * cycle_len

        # Build the full expected sequence (repeated cycles)
        full_sequence = played_bols * n_cycles

        # Map onsets to sequence positions
        result = []
        for i in range(onset_count):
            if onset_count <= len(full_sequence):
                # Map proportionally
                idx = int(i * len(full_sequence) / onset_count)
                idx = min(idx, len(full_sequence) - 1)
                result.append(full_sequence[idx])
            else:
                # More onsets than expected: cyclic wrap
                result.append(played_bols[i % cycle_len])

        return result

    def _bol_in_model_vocab(self, bol_name):
        """Check if a Bol from template is in our model's vocabulary."""
        return bol_name.lower() in self.BOL_CLASSES

    def predict_combined(self, audio):
        """Combined prediction: Template mapping + Model classification.
        
        Pipeline:
        1. Detect Taal from full audio
        2. Get template Bol pattern for that Taal
        3. Detect onsets (individual strokes)
        4. For each onset: classify with Bol model
        5. Align onsets to template pattern
        6. Cross-validate: compare template vs model predictions
        
        Returns timeline with both template and model predictions, plus
        a combined confidence-weighted result.
        """
        # Step 1: Detect Taal
        taal_result = self._classify(audio, "taal")
        taal_label = taal_result["label"]
        taal_conf = taal_result["confidence"]

        # Step 2: Get template
        pattern_info = self.TAAL_BOL_PATTERNS.get(taal_label, {})
        theka = pattern_info.get('theka', 'Unknown')

        # Step 3: Detect onsets
        onset_times = self._detect_onsets(audio)
        duration = len(audio) / self.sample_rate

        if len(onset_times) == 0:
            return {
                "taal": taal_result,
                "theka": theka,
                "bol_sequence": [],
                "bol_count": 0,
                "cycles_detected": 0,
            }

        # Step 4: Classify each segment with model
        model_predictions = []
        for i, onset in enumerate(onset_times):
            start_sample = int(onset * self.sample_rate)
            if i + 1 < len(onset_times):
                end_time = min(onset + 0.5, onset_times[i + 1])
            else:
                end_time = min(onset + 0.5, duration)
            end_sample = int(end_time * self.sample_rate)
            segment = audio[start_sample:end_sample]
            if len(segment) < int(0.03 * self.sample_rate):
                model_predictions.append(None)
                continue
            model_predictions.append(self._classify_segment(segment))

        # Step 5: Align onsets to template
        template_bols = self._align_onsets_to_template(len(onset_times), taal_label)
        played_bols = self._get_played_bols(taal_label)
        cycle_len = len(played_bols) if played_bols else 0
        n_cycles = max(1, round(len(onset_times) / cycle_len)) if cycle_len > 0 else 0

        # Step 6: Build combined timeline
        timeline = []
        for i, onset in enumerate(onset_times):
            end_t = onset_times[i + 1] if i + 1 < len(onset_times) else duration

            # Template prediction
            tmpl_bol = template_bols[i] if i < len(template_bols) else None

            # Model prediction
            model = model_predictions[i]
            model_bol = model["display_name"] if model else None
            model_conf = model["confidence"] if model else 0

            # Cross-validation: determine source and combined confidence
            if tmpl_bol and model:
                # Check if model agrees with template
                tmpl_lower = tmpl_bol.lower()
                model_lower = model["label"]

                if tmpl_lower == model_lower:
                    # Perfect agreement
                    source = "both"
                    combined_conf = min(1.0, taal_conf * 0.5 + model_conf * 0.5 + 0.2)
                    final_bol = tmpl_bol
                elif self._bol_in_model_vocab(tmpl_bol):
                    # Template Bol is in vocab but model disagrees
                    # Trust template if Taal confidence is high
                    if taal_conf > 0.6:
                        source = "template"
                        combined_conf = taal_conf * 0.7
                        final_bol = tmpl_bol
                    else:
                        source = "model"
                        combined_conf = model_conf
                        final_bol = model_bol
                else:
                    # Template Bol is NOT in model vocab (Na, Ge, Dhi, Ti, etc.)
                    # Template is authoritative
                    source = "template"
                    combined_conf = taal_conf * 0.8
                    final_bol = tmpl_bol
            elif tmpl_bol:
                source = "template"
                combined_conf = taal_conf * 0.7
                final_bol = tmpl_bol
            elif model:
                source = "model"
                combined_conf = model_conf
                final_bol = model_bol
            else:
                source = "unknown"
                combined_conf = 0
                final_bol = "?"

            # Beat position in cycle
            beat_in_cycle = i % cycle_len if cycle_len > 0 else i
            cycle_num = (i // cycle_len) + 1 if cycle_len > 0 else 1

            entry = {
                "time": round(float(onset), 3),
                "end_time": round(float(end_t), 3),
                "bol": final_bol,
                "confidence": round(combined_conf, 4),
                "source": source,
                "template_bol": tmpl_bol,
                "model_bol": model_bol,
                "model_confidence": round(model_conf, 4) if model else 0,
                "beat_position": beat_in_cycle + 1,
                "cycle": cycle_num,
            }
            # Add top 3 model predictions if available
            if model:
                entry["model_top_3"] = model["top_3"]

            timeline.append(entry)

        return {
            "taal": taal_result,
            "theka": theka,
            "bol_sequence": timeline,
            "bol_count": len(timeline),
            "cycles_detected": n_cycles,
            "pattern_length": cycle_len,
        }

    def predict_combined_from_file(self, file_path):
        """Combined Taal + Bol sequence prediction from file."""
        start_time = time.time()
        audio = self._load_audio(file_path)
        duration = len(audio) / self.sample_rate

        result = self.predict_combined(audio)
        result["status"] = "success"
        result["audio_info"] = {
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
            "source": os.path.basename(file_path) if isinstance(file_path, str) else "uploaded"
        }
        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)
        return result

    def predict_combined_from_bytes(self, audio_bytes):
        """Combined Taal + Bol sequence prediction from bytes."""
        start_time = time.time()
        audio = self._load_audio(audio_bytes)
        duration = len(audio) / self.sample_rate

        result = self.predict_combined(audio)
        result["status"] = "success"
        result["audio_info"] = {
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
            "source": "uploaded"
        }
        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)
        return result

    def get_model_info(self):
        """Return information about loaded models."""
        return {
            "bol_model": {
                "architecture": self.bol_model_name,
                "accuracy": round(self.bol_model_acc, 4),
                "classes": self.BOL_CLASSES,
                "input_duration_seconds": self.bol_max_duration,
            },
            "taal_model": {
                "architecture": self.taal_model_name,
                "accuracy": round(self.taal_model_acc, 4),
                "classes": self.TAAL_CLASSES,
                "input_duration_seconds": self.taal_max_duration,
            },
            "device": str(self.device),
            "sample_rate": self.sample_rate,
        }
