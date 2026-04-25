"""
Deep Learning Model Architectures for Tabla Audio Classification.

Four architectures designed for mel spectrogram classification:
1. ResidualCNN     - ResNet-inspired CNN with skip connections
2. CRNN            - CNN feature extractor + Bidirectional LSTM
3. AttentionCRNN   - CRNN with Multi-Head Self-Attention
4. ASTLite         - Lightweight Audio Spectrogram Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Building Blocks
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual convolutional block with optional downsampling."""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        # Shortcut connection for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out


class ConvBlock(nn.Module):
    """Standard convolutional block: Conv → BN → GELU → MaxPool → Dropout."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=(2, 2),
                 dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size) if pool_size else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.gelu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for sequence modeling."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


# =============================================================================
# Model 1: ResidualCNN
# =============================================================================

class ResidualCNN(nn.Module):
    """
    ResNet-inspired CNN for audio classification.
    
    Uses residual connections to enable deeper networks without degradation.
    Adaptive pooling makes it input-size agnostic.
    
    Architecture:
        Conv(1→32) → ResBlock(32→32) → ResBlock(32→64,s=2) →
        ResBlock(64→128,s=2) → ResBlock(128→256,s=2) →
        AdaptiveAvgPool → Dense(256) → Dropout → Dense(num_classes)
    """

    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Residual blocks with increasing channels
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )

        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Model 2: CRNN (Convolutional Recurrent Neural Network)
# =============================================================================

class CRNN(nn.Module):
    """
    CNN + Bidirectional LSTM for audio classification.
    
    CNN layers extract spectral features from mel spectrogram patches.
    BiLSTM captures temporal dependencies across the entire sequence.
    
    Architecture:
        ConvBlocks(1→32→64→128→128) → GlobalAvgPool(freq) →
        BiLSTM(128→256) → MeanPool(time) → Dense → num_classes
    """

    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, pool_size=(2, 2), dropout=0.1),     # → (32, 64, T/2)
            ConvBlock(32, 64, pool_size=(2, 2), dropout=0.1),    # → (64, 32, T/4)
            ConvBlock(64, 128, pool_size=(2, 2), dropout=0.15),  # → (128, 16, T/8)
            ConvBlock(128, 128, pool_size=(2, 1), dropout=0.15), # → (128, 8, T/8)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.layer_norm = nn.LayerNorm(256)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # (B, 128, H', T')

        # Global average pool over frequency dimension
        x = x.mean(dim=2)  # (B, 128, T')

        # Reshape for LSTM: (B, T', 128)
        x = x.permute(0, 2, 1)

        # BiLSTM
        x, _ = self.lstm(x)  # (B, T', 256)
        x = self.layer_norm(x)

        # Mean pooling over time
        x = x.mean(dim=1)  # (B, 256)

        # Classify
        x = self.classifier(x)
        return x


# =============================================================================
# Model 3: AttentionCRNN (CRNN + Multi-Head Self-Attention)
# =============================================================================

class AttentionCRNN(nn.Module):
    """
    CNN + BiLSTM + Multi-Head Self-Attention for audio classification.
    
    Extends CRNN with an attention mechanism that lets the model focus
    on the most discriminative time segments in the audio.
    
    Architecture:
        ConvBlocks(1→32→64→128→128) → GlobalAvgPool(freq) →
        BiLSTM(128→256) → MultiHeadAttention(4 heads) →
        WeightedPool(time) → Dense → num_classes
    """

    def __init__(self, num_classes=8, dropout=0.3, n_heads=4):
        super().__init__()

        # CNN feature extractor (same as CRNN)
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, pool_size=(2, 2), dropout=0.1),
            ConvBlock(32, 64, pool_size=(2, 2), dropout=0.1),
            ConvBlock(64, 128, pool_size=(2, 2), dropout=0.15),
            ConvBlock(128, 128, pool_size=(2, 1), dropout=0.15),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Multi-Head Self-Attention
        self.attention = MultiHeadAttention(256, n_heads=n_heads, dropout=0.1)

        # Feed-forward after attention
        self.ff = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.Dropout(0.1)
        )
        self.ff_norm = nn.LayerNorm(256)

        # Learnable attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(256, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN
        x = self.cnn(x)
        x = x.mean(dim=2)          # Global avg pool over frequency
        x = x.permute(0, 2, 1)     # (B, T', 128)

        # BiLSTM
        x, _ = self.lstm(x)        # (B, T', 256)

        # Self-Attention
        x = self.attention(x)      # (B, T', 256)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)

        # Attention pooling: learn which time steps matter most
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # (B, T', 1)
        x = (x * attn_weights).sum(dim=1)  # (B, 256)

        # Classify
        x = self.classifier(x)
        return x


# =============================================================================
# Model 4: ASTLite (Lightweight Audio Spectrogram Transformer)
# =============================================================================

class ASTLite(nn.Module):
    """
    Lightweight Audio Spectrogram Transformer.
    
    Treats the mel spectrogram as an image, divides it into patches,
    and processes them with a Transformer encoder. Inspired by Vision
    Transformer (ViT) and AST (Audio Spectrogram Transformer).
    
    Architecture:
        PatchEmbed(16×16) → PositionEmbed → CLS token →
        TransformerEncoder(×4, d=192, 4 heads) →
        CLS output → Dense → num_classes
    """

    def __init__(self, num_classes=8, d_model=192, n_heads=4,
                 num_layers=4, patch_size=16, max_patches=300, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # Patch embedding via Conv2d
        self.patch_embed = nn.Conv2d(
            1, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_patches + 1, d_model) * 0.02
        )

        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True      # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        B = x.shape[0]

        # Create patches: (B, 1, n_mels, T) → (B, d_model, H', W')
        patches = self.patch_embed(x)
        H, W = patches.shape[2], patches.shape[3]

        # Flatten spatial dims: (B, d_model, H'*W') → (B, n_patches, d_model)
        patches = patches.flatten(2).transpose(1, 2)
        n_patches = patches.shape[1]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)  # (B, 1+n_patches, d)

        # Add position embeddings (truncated to actual sequence length)
        patches = patches + self.pos_embed[:, :1 + n_patches, :]
        patches = self.pos_drop(patches)

        # Transformer encoder
        out = self.transformer(patches)  # (B, 1+n_patches, d)

        # Classification from CLS token
        cls_out = out[:, 0]  # (B, d)

        return self.head(cls_out)


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    "ResidualCNN": ResidualCNN,
    "CRNN": CRNN,
    "AttentionCRNN": AttentionCRNN,
    "ASTLite": ASTLite,
}


def get_model(name: str, num_classes: int = 8, **kwargs) -> nn.Module:
    """Factory function to create a model by name.
    
    Args:
        name: model name from MODEL_REGISTRY
        num_classes: number of output classes
        **kwargs: additional model-specific arguments
    
    Returns:
        nn.Module instance
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, name: str = ""):
    """Print a brief model summary."""
    n_params = count_parameters(model)
    if n_params >= 1_000_000:
        param_str = f"{n_params / 1_000_000:.2f}M"
    else:
        param_str = f"{n_params / 1_000:.1f}K"
    print(f"  {name or model.__class__.__name__}: {param_str} parameters")
    return n_params
