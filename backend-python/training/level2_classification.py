import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with pre-norm architecture.
    
    Prevents gradient degradation in deeper networks and allows
    the model to learn identity mappings when a layer isn't helpful,
    which is critical for small datasets where over-parameterization
    can hurt.
    """
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Level2Classifier(nn.Module):
    """
    Level 2 — Lightweight Real-Time Models (Neural Classification)
    
    This head attaches directly to the Level 1 extraction embeddings 
    (e.g., 512-dim from Whisper Base) and projects them to actionable clinical metrics:
    1. A 3-class emotion categorization: Calm, Mild Anxiety, High Anxiety
    2. A continuous arousal regression score (1.0 to 10.0 scale)
    
    Improvements over v1:
    - Deeper network with residual connections (prevents gradient vanishing)
    - GELU activation (smoother gradients than ReLU, better for emotion tasks)
    - LayerNorm instead of BatchNorm (stable with batch_size=1 during inference)
    - Reduced dropout (0.25 vs 0.40 — original was too aggressive for this task)
    - Separate deeper heads for emotion and arousal (multi-task benefits)
    """
    def __init__(self, input_dim=512, hidden_dim=256, dropout_rate=0.25):
        super().__init__()
        self.input_dim = input_dim
        
        # Shared backbone with residual connections
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        self.shared_residual = nn.Sequential(
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate),
        )
        
        # Branch 1: 3-Class Emotion Categorization (Calm, Mild, High)
        # Deeper head for better class separation
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Branch 2: Continuous Arousal Regressor (1-10 scale)
        # Separate deeper head for regression task
        self.arousal_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # We will scale this to 1-10 in the forward pass
        )

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embedding (torch.Tensor): Feature vector of shape [Batch, InputDim].
                                      Typically extracted from Level1FeatureExtractor.
        
        Returns:
            emotion_logits (torch.Tensor): Raw classification logits [Batch, 3].
            arousal_score (torch.Tensor): Continuous score from 1.0 to 10.0 [Batch, 1].
        """
        # Handle 1D inputs by artificially injecting a batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        # Shared backbone
        h = self.input_proj(embedding)
        h = self.shared_residual(h)
        
        # 3-class logits
        emotion_logits = self.emotion_classifier(h)
        
        # Arousal score (0.0 to 1.0 mapped to 1.0 to 10.0)
        arousal_norm = self.arousal_regressor(h)
        arousal_score = 1.0 + (arousal_norm * 9.0)
        
        return emotion_logits, arousal_score


if __name__ == "__main__":
    print("Level 2 Classification module loaded successfully.")
    print("This file contains the Level2Classifier PyTorch model.")
    print("Improvements: Residual blocks, GELU, LayerNorm, deeper heads.")
    print("To test the full classification inference, run: python test_level2.py")
