import numpy as np
import time
import torch

from level1_foundation import Level1FeatureExtractor
from level2_classification import Level2Classifier

def run_test():
    print("=" * 60)
    print(" AuraOS: Testing Level 1 + Level 2 Deep Inference Pipeline ")
    print("=" * 60)
    print("Improvements: Residual blocks, GELU, LayerNorm, deeper heads")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    try:
        # Load Phase 1
        print("\nLoading Level 1 (Whisper Foundation Extractor v2)...")
        extractor = Level1FeatureExtractor(model_type="whisper")
        
        # Load Phase 2 (Assuming embedding size is 512 for whisper-base)
        print("Loading Level 2 (Residual Classification Head v2)...")
        classifier = Level2Classifier(input_dim=512)
        classifier.to(device)
        classifier.eval()
        
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # Simulate 5-second raw audio array from client at 16kHz
    print("\nSimulating 5.0 seconds of audio data (16kHz)...")
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32)

    # Warm-up inference usually needed for torch/cuda
    print("Running initial warm-up pass...")
    with torch.no_grad():
        warmup_features = extractor(dummy_audio)
        _, _ = classifier(warmup_features)

    # Benchmark Speed
    print("Benchmarking true inference latency...")
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # LEVEL 1: Map wave -> Embedding  [Batch, 512]
            features = extractor(dummy_audio)
            
            # LEVEL 2: Map Embedding -> [Batch, 3] and [Batch, 1]
            emotion_logits, arousal_score = classifier(features)
            
            # Post-process Logits to Probabilities
            emotion_probs = torch.nn.functional.softmax(emotion_logits, dim=-1)
            
        dt = (time.time() - start_time) * 1000 # Convert to milliseconds
        
        print("\n" + "-" * 60)
        print(f"SUCCESS! Fully chained inference complete in {dt:.2f} ms")
        print("-" * 60)
        
        print(f"\n[Level 1 Output shape]: {features.shape}")
        print(f"[Level 2 Emotion Probabilities]: {emotion_probs.cpu().numpy()[0]}")
        print(f"[Level 2 Arousal Score (1-10)]: {arousal_score.cpu().item():.2f}")
        
        # Validate outputs
        assert features.shape[-1] == 512, f"Expected 512-dim features, got {features.shape[-1]}"
        assert emotion_logits.shape[-1] == 3, f"Expected 3-class logits, got {emotion_logits.shape[-1]}"
        assert 1.0 <= arousal_score.item() <= 10.0, f"Arousal out of range: {arousal_score.item()}"
        print("\n[PASS] All output dimensions and ranges verified!")
        
    except Exception as e:
        print(f"Error during forward pass execution: {e}")
        
    finally:
        # Avoid OOM
        del extractor, classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_test()
