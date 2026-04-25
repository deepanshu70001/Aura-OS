import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Self-attention layer for temporal arousal sequences.
    
    Allows the model to weigh which time-steps in the arousal history
    are most predictive of escalation, rather than treating all
    time-steps equally through the LSTM hidden state.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim // 2)
        self.key = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = (hidden_dim // 2) ** 0.5
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: [Batch, SeqLen, HiddenDim*2] from bidirectional LSTM
        Returns:
            context: [Batch, HiddenDim*2] attention-weighted context
        """
        Q = self.query(lstm_output)   # [B, T, H/2]
        K = self.key(lstm_output)     # [B, T, H/2]
        V = self.value(lstm_output)   # [B, T, H]
        
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, V)  # [B, T, H]
        
        # Pool over time dimension
        return context.mean(dim=1)  # [B, H]


class TemporalContextLSTM(nn.Module):
    """
    Stage 3 — Temporal Context (Bi-LSTM + Self-Attention)
    
    Examines a rolling window of recent numerical arousal scores to detect 
    true escalation patterns over time, differentiating a panic pattern 
    (e.g., 4.0 -> 6.5 -> 8.2) from an isolated joyful shout (4.0 -> 9.0 -> 4.5).
    
    Improvements over v1:
    - Self-attention mechanism over LSTM outputs (learns which time-steps matter)
    - Deeper classifier head with residual path
    - Gradient-friendly architecture with LayerNorm
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        # 2-layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Self-attention over LSTM outputs
        self.temporal_attention = TemporalAttention(hidden_dim * 2)
        
        # Output layer maps attention context to binary probability
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, arousal_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            arousal_sequence (torch.Tensor): A sequence of scores.
                                             Expected shape: [Batch, SequenceLength, 1]
                                             e.g., [1, 6, 1] for 30 seconds of 5-sec slices.
        Returns:
            torch.Tensor: Escalation probability [Batch, 1], constrained 0.0-1.0
        """
        # Ensure correct shape
        if arousal_sequence.dim() == 2:
            arousal_sequence = arousal_sequence.unsqueeze(-1)
            
        lstm_out, (h_n, c_n) = self.lstm(arousal_sequence)  # [B, T, H*2]
        
        # Attention-weighted context (better than just using final hidden state)
        context = self.temporal_attention(lstm_out)  # [B, H*2]
        
        escalation_prob = self.classifier(context)
        return escalation_prob


def final_decision_gate(
    stage1_physiology: bool, 
    stage2_arousal_score: float, 
    stage3_escalation_prob: float, 
    escalation_threshold: float = 0.5
) -> dict:
    """
    The Final Safe-Net Decision Layer.
    
    A clinical alert fires STRICTLY when:
      Stage 1 physiological signal = TRUE
      AND Stage 2 arousal score > 7.5
      AND (Stage 3 escalation detected OR single reading > 9.0)
    
    Args:
        stage1_physiology (bool): From AcousticPhysicsExtractor (jitter/shimmer limits tripped)
        stage2_arousal_score (float): From Level2Classifier (1.0 to 10.0 scale)
        stage3_escalation_prob (float): From TemporalContextLSTM outputs (0.0 to 1.0)
        
    Returns:
        dict: Triage response detailing pass/fail logic
    """
    # Evaluate individual constraints
    physiology_check = bool(stage1_physiology)
    arousal_check = (stage2_arousal_score > 7.5)
    
    is_escalating = (stage3_escalation_prob > escalation_threshold)
    is_extreme_spike = (stage2_arousal_score > 9.0)
    
    temporal_check = (is_escalating or is_extreme_spike)
    
    # Final Clinical determination
    alert_triggered = physiology_check and arousal_check and temporal_check
    
    return {
        "alert_triggered": alert_triggered,
        "diagnostics": {
            "physiology_check_passed": physiology_check,
            "arousal_check_passed": arousal_check,
            "temporal_check_passed": temporal_check,
            "escalating_pattern": is_escalating,
            "extreme_spike": is_extreme_spike
        }
    }


if __name__ == "__main__":
    print("Level 4 Temporal Context & Decision Gate module loaded.")
    print("Improvements: Self-attention over LSTM outputs, deeper classifier, LayerNorm.")
    print("To test the logic boundaries and execution speed, run: python test_level4.py")
