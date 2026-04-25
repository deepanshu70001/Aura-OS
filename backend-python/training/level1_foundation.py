import torch
import torch.nn as nn
from transformers import (
    WhisperModel, 
    WhisperFeatureExtractor,
    WavLMModel,
    Wav2Vec2FeatureExtractor
)
import librosa
import numpy as np

# Global device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over temporal dimension.
    
    Instead of naive mean-pooling which treats all time-steps equally,
    this learns which time-steps carry the most emotional signal.
    Critical for speech emotion: stress markers concentrate in specific
    phonemes, not spread uniformly across the utterance.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1, bias=False),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [Batch, Time, HiddenDim]
        Returns:
            pooled: [Batch, HiddenDim]
        """
        # Compute attention weights over time dimension
        attn_weights = self.attention(hidden_states)          # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)    # [B, T, 1]
        # Weighted sum
        pooled = torch.sum(hidden_states * attn_weights, dim=1)  # [B, HiddenDim]
        return pooled


class Level1FeatureExtractor(nn.Module):
    """
    Level 1 — Foundation Models (Feature Extractor)
    
    This module encapsulates a chosen heavy pre-trained audio model (Whisper or WavLM).
    These models are optimized to extract rich, contextual embeddings from audio without 
    running full transcription or decoding layers.
    
    Improvements over v1:
    - Attention-weighted pooling instead of mean pooling (learns which frames matter)
    - Multi-scale feature extraction (concat mean + std + attention for richer representation)
    """
    def __init__(self, model_type="whisper"):
        super().__init__()
        self.model_type = model_type.lower()
        
        # Load models and extractors appropriately 
        if self.model_type == "whisper":
            # Using the `base` model to fit safely inside 4GB VRAM.
            model_id = "openai/whisper-base"
            self.processor = WhisperFeatureExtractor.from_pretrained(model_id)
            # We use WhisperModel which includes encoder/decoder, 
            # but we will only run the encoder in the forward pass.
            self.model = WhisperModel.from_pretrained(model_id).encoder
            hidden_dim = 512
            
        elif self.model_type == "wavlm":
            model_id = "microsoft/wavlm-base-plus"
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = WavLMModel.from_pretrained(model_id)
            hidden_dim = 768
            
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
            
        # Freeze all parameters! This is a feature extractor only.
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        self.model.to(DEVICE)
        
        # Attention pooling layer (trainable, lightweight)
        self.attention_pool = AttentionPooling(hidden_dim)
        self.attention_pool.to(DEVICE)
        
        # Output projection: concat(mean_pool, std_pool, attn_pool) -> hidden_dim
        # This gives 3x richer representation that captures:
        #   - mean: overall spectral characteristics
        #   - std: variability/volatility in the signal
        #   - attention: most emotionally salient frames
        self.output_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_proj.to(DEVICE)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_norm.to(DEVICE)
        
    @torch.no_grad()
    def forward(self, audio_array: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """
        Takes raw audio, processes it through the feature extractor, and
        returns pooled representation from the foundation model.
        
        Args:
            audio_array (np.ndarray): 1D audio waveform.
            sr (int): Sample rate of the audio (default 16kHz).
            
        Returns:
            torch.Tensor: Pooled representation [1, hidden_dim].
        """
        # Ensure correct sample rate (16kHz is expected for these models)
        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
        if self.model_type == "whisper":
            # Whisper uses a log-Mel spectrogram internally through its feature extractor
            inputs = self.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(DEVICE)
            outputs = self.model(input_features)
            
        elif self.model_type == "wavlm":
            # WavLM expects raw waveform values
            inputs = self.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(DEVICE)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(DEVICE)
                
            outputs = self.model(input_values, attention_mask=attention_mask)
            
        # Multi-scale pooling for richer features
        last_hidden_state = outputs.last_hidden_state  # [Batch, Time, HiddenDim]
        
        mean_pool = torch.mean(last_hidden_state, dim=1)          # [B, H]
        std_pool = torch.std(last_hidden_state, dim=1)             # [B, H]
        attn_pool = self.attention_pool(last_hidden_state)         # [B, H]
        
        # Concatenate all three pooling strategies
        multi_scale = torch.cat([mean_pool, std_pool, attn_pool], dim=-1)  # [B, 3*H]
        
        # Project back to original dim with layer norm
        pooled_features = self.output_norm(self.output_proj(multi_scale))  # [B, H]
        
        return pooled_features

if __name__ == "__main__":
    print("Level 1 Foundation module loaded successfully.")
    print("This file contains the Level1FeatureExtractor class.")
    print("Improvements: Attention-weighted pooling + multi-scale features.")
    print("To test the extractor, run: python test_level1.py")
