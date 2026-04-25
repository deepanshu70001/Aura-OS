import numpy as np
import parselmouth
from parselmouth.praat import call

class AcousticPhysicsExtractor:
    """
    Level 3 / Stage 1: Acoustic Physics (Physics-First Approach)
    
    This module uses Praat bindings to detect physical vocal cord tremors and 
    biomarkers of stress, completely bypassing neural networks. 
    It operates as a strict physiological gate.
    """
    
    def __init__(self, jitter_threshold=0.025, shimmer_threshold=0.035):
        """
        Args:
            jitter_threshold (float): Threshold for pitch perturbation (default 2.5%)
            shimmer_threshold (float): Threshold for amplitude perturbation (default 3.5%)
        """
        self.jitter_threshold = jitter_threshold
        self.shimmer_threshold = shimmer_threshold

    def extract_biomarkers(self, audio_array: np.ndarray, sr: int = 16000) -> dict:
        """
        Directly injects the Numpy array into Praat's C++ memory buffer.
        Extracts vocal physiological biomarkers continuously.
        
        Args:
            audio_array (np.ndarray): 1D audio waveform.
            sr (int): Sample rate
            
        Returns:
            dict: {
                'jitter': float,
                'shimmer': float,
                'hnr': float,
                'f0_mean': float,
                'physiologically_stressed': bool
            }
        """
        # Bypassing disk I/O entirely: create Sound object from memory
        # Parselmouth expects shape (channels, time), so we reshape 1D to (1, time)
        if audio_array.ndim == 1:
            snd_data = audio_array.reshape(1, -1)
        else:
            snd_data = audio_array
            
        snd = parselmouth.Sound(snd_data, sampling_frequency=sr)
        
        # 1. Fundamental Frequency (F0)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency'].copy()
        # Ignore unvoiced segments (F0 = 0) — must copy first to avoid mutating Praat internals
        pitch_values[pitch_values == 0] = np.nan
        f0_mean = np.nanmean(pitch_values) if not np.all(np.isnan(pitch_values)) else 0.0
        
        # 2. Point Process for Jitter/Shimmer
        # (cross-correlation method usually yields the highest clinical detail)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        
        # Jitter (local): 0.0001 to 0.02 is Praat's recommended period range, 1.3 is max period factor
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Shimmer (local)
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # 3. Harmonics-to-Noise Ratio (HNR)
        hnr = call(snd.to_harmonicity(), "Get mean", 0, 0)
        
        # Clean up NaNs which happen if Praat couldn't detect voice bounds
        jitter = 0.0 if np.isnan(jitter) else float(jitter)
        shimmer = 0.0 if np.isnan(shimmer) else float(shimmer)
        hnr = 0.0 if np.isnan(hnr) else float(hnr)
        f0_mean = 0.0 if np.isnan(f0_mean) else float(f0_mean)

        # 4. Strict Decision Gate (No AI hallucination allowed here)
        # Panic attacks physically alter the harmonic-to-noise ratio and inject micro-tremors 
        # (high jitter) into the vocal cords.
        physiologically_stressed = (jitter > self.jitter_threshold) and (shimmer > self.shimmer_threshold)
        
        return {
            'jitter': jitter,
            'shimmer': shimmer,
            'hnr': hnr,
            'f0_mean': f0_mean,
            'physiologically_stressed': physiologically_stressed
        }

if __name__ == "__main__":
    print("Level 3 Acoustic Physics module loaded.")
    print("This file contains the AcousticPhysicsExtractor class.")
    print("To benchmark the <5ms requirement, run: python test_level3.py")
