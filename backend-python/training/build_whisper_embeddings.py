import os
import re
import argparse
import numpy as np
import torch
import librosa
from tqdm import tqdm
from level1_foundation import Level1FeatureExtractor

# Configuration
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
TESTING_ROOT = os.path.join(os.path.dirname(__file__), "testing", "Audio_Dataset")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "processed_whisper_embeddings.npz")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
DURATION = 5.0

# ---------------------------------------------------------------
# Clinical 3-class mapping from legacy 8-class emotion labels
# ---------------------------------------------------------------
CLINICAL_MAP = {
    'neutral': 'calm',
    'calm': 'calm',
    'happy': 'calm',
    'surprise': 'calm',
    'sad': 'mild_anxiety',
    'disgust': 'mild_anxiety',
    'fear': 'high_anxiety',
    'angry': 'high_anxiety',
}

# ---------------------------------------------------------------
# Dataset parsers (ported from the original build_dataset.py)
# ---------------------------------------------------------------

def parse_iemocap():
    """Parse IEMOCAP dataset using emotion evaluation text files."""
    processed = []
    EMOTION_MAP = {
        'neu': 'neutral', 'hap': 'happy', 'exc': 'happy',
        'sad': 'sad', 'ang': 'angry', 'fru': 'angry',
        'fea': 'fear', 'dis': 'disgust', 'sur': 'surprise',
    }
    iemocap_root = os.path.join(DATASETS_DIR, "IEMOCAP_full_release")
    if not os.path.exists(iemocap_root):
        return processed
    for sess in [f"Session{i}" for i in range(1, 6)]:
        emo_eval_dir = os.path.join(iemocap_root, sess, "dialog", "EmoEvaluation")
        if not os.path.exists(emo_eval_dir):
            continue
        for file in os.listdir(emo_eval_dir):
            if not file.endswith(".txt"):
                continue
            with open(os.path.join(emo_eval_dir, file), 'r') as f:
                for line in f:
                    if line.startswith("["):
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            turn_name, emo_code = parts[1], parts[2]
                            if emo_code in EMOTION_MAP:
                                subfolder = turn_name[:-5]
                                wav_path = os.path.join(iemocap_root, sess, "sentences", "wav", subfolder, f"{turn_name}.wav")
                                if os.path.exists(wav_path):
                                    clinical = CLINICAL_MAP.get(EMOTION_MAP[emo_code])
                                    if clinical:
                                        processed.append((wav_path, clinical))
    return processed

def parse_ravdess():
    """Parse RAVDESS dataset using filename encoding."""
    processed = []
    root = os.path.join(DATASETS_DIR, "ravdess")
    if not os.path.exists(root):
        return processed
    rav_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise',
    }
    for actor in os.listdir(root):
        actor_dir = os.path.join(root, actor)
        if not os.path.isdir(actor_dir):
            continue
        for file in os.listdir(actor_dir):
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) >= 3:
                    code = parts[2]
                    raw_label = rav_map.get(code)
                    if raw_label:
                        clinical = CLINICAL_MAP.get(raw_label)
                        if clinical:
                            processed.append((os.path.join(actor_dir, file), clinical))
    return processed

def parse_tess():
    """Parse TESS dataset using folder name encoding."""
    processed = []
    root = os.path.join(DATASETS_DIR, "tess")
    if not os.path.exists(root):
        return processed
    for folder in os.listdir(root):
        emotion = folder.split('_')[-1].lower()
        clinical = CLINICAL_MAP.get(emotion)
        if clinical:
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    processed.append((os.path.join(folder_path, file), clinical))
    return processed

def parse_crema_d():
    """Parse CREMA-D dataset using filename encoding."""
    processed = []
    root = os.path.join(DATASETS_DIR, "crema-d")
    if not os.path.exists(root):
        return processed
    cmap = {'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'}
    for file in os.listdir(root):
        if file.endswith(".wav"):
            parts = file.split('_')
            if len(parts) >= 3:
                code = parts[2]
                raw_label = cmap.get(code)
                if raw_label:
                    clinical = CLINICAL_MAP.get(raw_label)
                    if clinical:
                        processed.append((os.path.join(root, file), clinical))
    return processed

def parse_savee():
    """Parse SAVEE dataset using filename prefix encoding."""
    processed = []
    root = os.path.join(DATASETS_DIR, "Savee")
    if not os.path.exists(root):
        return processed
    smap = {'a': 'angry', 'h': 'happy', 'sa': 'sad', 'n': 'neutral', 'f': 'fear', 'd': 'disgust', 'su': 'surprise'}
    for file in os.listdir(root):
        if file.endswith(".wav"):
            matches = re.findall(r'[a-z]+', file.split('_')[-1])
            if matches:
                code = matches[0]
                raw_label = smap.get(code)
                if raw_label:
                    clinical = CLINICAL_MAP.get(raw_label)
                    if clinical:
                        processed.append((os.path.join(root, file), clinical))
    return processed

def parse_jl_corpus():
    """Parse JL Corpus dataset using filename encoding."""
    processed = []
    root = os.path.join(DATASETS_DIR, "JL Corpus (wav+txt)")
    if not os.path.exists(root):
        return processed
    for file in os.listdir(root):
        if file.endswith(".wav"):
            parts = file.split('_')
            if len(parts) >= 2:
                emotion = parts[1].lower()
                clinical = CLINICAL_MAP.get(emotion)
                if clinical:
                    processed.append((os.path.join(root, file), clinical))
    return processed

def parse_custom_test():
    """Parse custom testing audio dataset."""
    processed = []
    if not os.path.exists(TESTING_ROOT):
        return processed
    for folder in os.listdir(TESTING_ROOT):
        clinical = CLINICAL_MAP.get(folder.lower())
        if clinical:
            folder_path = os.path.join(TESTING_ROOT, folder)
            if not os.path.isdir(folder_path):
                continue
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    processed.append((os.path.join(folder_path, file), clinical))
    return processed

# ---------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------

def collect_all_files(include_custom_test: bool = False):
    """Aggregate files from dataset sources."""
    all_files = []
    parsers = [
        ("IEMOCAP", parse_iemocap),
        ("RAVDESS", parse_ravdess),
        ("TESS", parse_tess),
        ("CREMA-D", parse_crema_d),
        ("SAVEE", parse_savee),
        ("JL Corpus", parse_jl_corpus),
    ]
    if include_custom_test:
        parsers.append(("Custom Test", parse_custom_test))

    for name, parser in parsers:
        files = parser()
        print(f"  {name}: {len(files)} files")
        all_files.extend(files)

    deduped = []
    seen = set()
    duplicate_count = 0
    for wav_path, label in all_files:
        key = os.path.normcase(os.path.normpath(wav_path))
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append((wav_path, label))

    if duplicate_count:
        print(f"  [WARN] Removed {duplicate_count} duplicate file entries")

    return deduped


def _print_class_distribution(labels: list[str]) -> None:
    if not labels:
        return
    unique, counts = np.unique(np.array(labels), return_counts=True)
    dist = {str(k): int(v) for k, v in zip(unique, counts)}
    print(f"Class distribution: {dist}")


def build_cache(include_custom_test: bool = False, allow_dummy: bool = False):
    print("=" * 60)
    print(" AuraOS: Building Whisper Embeddings (Full Dataset) ")
    print("=" * 60)

    files_to_process = collect_all_files(include_custom_test=include_custom_test)
    total = len(files_to_process)

    if total == 0:
        print("WARN: No audio files found in any dataset folder.")
        if not allow_dummy:
            raise RuntimeError(
                "No audio found. Refusing to generate dummy embeddings. "
                "Provide datasets or pass --allow-dummy for local smoke tests."
            )
        print("Injecting deterministic dummy cache for smoke testing only...")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 512), dtype=np.float32)
        y = np.array(["calm", "mild_anxiety", "high_anxiety"] * 16 + ["calm", "mild_anxiety"])
        rms = rng.uniform(0.01, 0.08, 50).astype(np.float32)
        np.savez_compressed(OUTPUT_PATH, X=X, y=y, rms=rms)
        print(f"[WARN] Dummy npz cached at {OUTPUT_PATH}")
        return

    print(f"\nTotal: {total} audio files across all datasets.")
    print(f"Loading Level 1 Whisper Pipeline on {DEVICE}...")

    extractor = Level1FeatureExtractor(model_type="whisper")
    extractor.eval()

    X_embeds = []
    y_labels = []
    rms_values = []
    skipped = 0
    failed_files = []

    print("Starting extraction...\n")
    with torch.no_grad():
        for file_path, label in tqdm(files_to_process, desc="Processing"):
            try:
                audio, sr = librosa.load(file_path, sr=SR, duration=DURATION)

                # Pad/truncate to exactly DURATION seconds
                target_len = int(SR * DURATION)
                if len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))
                else:
                    audio = audio[:target_len]

                # Extract real RMS energy (used as arousal ground-truth proxy)
                rms_val = float(np.sqrt(np.mean(np.square(audio))))

                features = extractor(audio)  # [1, 512]
                X_embeds.append(features.squeeze(0).cpu().numpy())
                y_labels.append(label)
                rms_values.append(rms_val)

            except Exception as e:
                skipped += 1
                if len(failed_files) < 15:
                    failed_files.append((file_path, str(e)))

    print(f"\nExtracted: {len(X_embeds)} | Skipped: {skipped}")
    if failed_files:
        print("First extraction failures:")
        for path, err in failed_files[:5]:
            print(f"  - {path} -> {err}")

    if not X_embeds:
        raise RuntimeError("Embedding extraction failed for all files; no cache written.")

    _print_class_distribution(y_labels)
    np.savez_compressed(
        OUTPUT_PATH,
        X=np.array(X_embeds),
        y=np.array(y_labels),
        rms=np.array(rms_values, dtype=np.float32),
    )
    print(f"SUCCESS! Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Whisper embedding cache for AuraOS.")
    parser.add_argument(
        "--include-custom-test",
        action="store_true",
        help="Include files under training/testing/Audio_Dataset in cache generation.",
    )
    parser.add_argument(
        "--allow-dummy",
        action="store_true",
        help="Allow deterministic dummy cache generation when no data is found (smoke tests only).",
    )
    args = parser.parse_args()
    build_cache(include_custom_test=args.include_custom_test, allow_dummy=args.allow_dummy)
