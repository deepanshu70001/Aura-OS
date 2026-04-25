import numpy as np
import time
from level1_foundation import Level1FeatureExtractor
import torch

def run_test():
    print("=" * 60)
    print(" AuraOS: Testing Level 1 Foundation Extractor (v2) ")
    print("=" * 60)
    print("Improvements: Attention-weighted pooling + multi-scale features")
    
    print("\nInitializing Level 1 Foundation Extractor (Whisper-Base)...")
    try:
        extractor = Level1FeatureExtractor(model_type="whisper")
        print("Success! Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy 5-second audio at 16kHz
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32)

    # Warm-up pass (first pass loads CUDA kernels)
    print("\nRunning warm-up pass...")
    with torch.no_grad():
        _ = extractor(dummy_audio)

    print("Running benchmarked forward pass (extracting features)...")
    start_time = time.time()
    try:
        with torch.no_grad():
            features = extractor(dummy_audio)
        dt = (time.time() - start_time) * 1000
        print(f"Success! Forward pass completed in {dt:.2f} ms")
        print(f"Feature tensor shape: {features.shape}")
        print(f"Feature stats: mean={features.mean().item():.4f}, std={features.std().item():.4f}")
        
        # Verify output dimension matches expected 512
        assert features.shape[-1] == 512, f"Expected 512-dim output, got {features.shape[-1]}"
        print("[PASS] Output dimension verified: 512")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        
    # Free cache to avoid any OOM issues on small GPUs
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nTest finished and memory cleared.")

if __name__ == "__main__":
    run_test()
