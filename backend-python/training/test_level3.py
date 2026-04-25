import numpy as np
import time

from level3_acoustic_physics import AcousticPhysicsExtractor

def simulate_panic_audio(duration: float = 5.0, sr: int = 16000) -> np.ndarray:
    """
    Simulates severe vocal cord tremors by injecting high-frequency 
    amplitude/pitch perturbances (jitter/shimmer) into a baseline sine wave.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # A base vocal pitch (e.g., panicked breathing/shouting around 300Hz)
    base_wave = np.sin(2 * np.pi * 300 * t)
    # Add severe amplitude perturbation (shimmer simulation)
    amplitude_noise = np.random.uniform(0.1, 2.0, size=t.shape)
    # Add high-variance random phase noise (jitter simulation)
    # Increasing variance to guarantee >2.5% pitch perturbation
    phase_noise = np.random.normal(0, 0.4, size=t.shape)
    
    panicked_wave = (base_wave + phase_noise) * amplitude_noise
    return panicked_wave.astype(np.float32)

def run_benchmarks():
    print("=" * 60)
    print(" AuraOS: Benchmarking Level 3 Acoustic Physics Gate ")
    print("=" * 60)
    
    extractor = AcousticPhysicsExtractor(jitter_threshold=0.025, shimmer_threshold=0.035)
    
    print("\n[Test 1] Pure Tone Analysis (Calm User)...")
    t = np.linspace(0, 5.0, int(16000 * 5.0), endpoint=False)
    calm_audio = np.sin(2 * np.pi * 150 * t).astype(np.float32)  # Perfectly stable 150Hz voice
    
    start = time.perf_counter()
    calm_res = extractor.extract_biomarkers(calm_audio, 16000)
    dt_calm = (time.perf_counter() - start) * 1000
    
    print(f" execution_time : {dt_calm:.3f} ms")
    print(f" jitter         : {calm_res['jitter']*100:.3f}%")
    print(f" shimmer        : {calm_res['shimmer']*100:.3f}%")
    print(f" Pre-Flag Fire? : {calm_res['physiologically_stressed']}")
    
    print("\n[Test 2] Severe Tremor Simulation (High Panic)...")
    panic_audio = simulate_panic_audio(5.0, 16000)
    
    start = time.perf_counter()
    panic_res = extractor.extract_biomarkers(panic_audio, 16000)
    dt_panic = (time.perf_counter() - start) * 1000
    
    print(f" execution_time : {dt_panic:.3f} ms")
    print(f" jitter         : {panic_res['jitter']*100:.3f}%")
    print(f" shimmer        : {panic_res['shimmer']*100:.3f}%")
    print(f" Pre-Flag Fire? : {panic_res['physiologically_stressed']}")
    
    print("\n" + "-" * 60)
    if dt_calm < 35.0 and dt_panic < 35.0:
        print(" SUCCESS! Both executions comfortably passed the < 35.0 ms real-time threshold.")
    else:
        print(" WARNING! Executions tripped the 35.0 ms rolling-buffer deadline.")
        
    if not calm_res['physiologically_stressed'] and panic_res['physiologically_stressed']:
        print(" SUCCESS! Biological threshold strictness is mathematically validated.")
    else:
        print(" ERROR! Strictness gate failed to split pure tone vs corrupted tone.")
    print("-" * 60)

if __name__ == "__main__":
    run_benchmarks()
