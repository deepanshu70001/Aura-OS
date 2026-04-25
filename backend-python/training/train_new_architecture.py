import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from level2_classification import Level2Classifier
from level4_temporal_gate import TemporalContextLSTM

EMBEDS_FILE = os.path.join(os.path.dirname(__file__), "processed_whisper_embeddings.npz")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

LABEL_MAP = {"calm": 0, "mild_anxiety": 1, "high_anxiety": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# ---------------------------------------------------------------
# Derive arousal from RMS energy + class prior
# ---------------------------------------------------------------
# RMS typically ranges 0.001 (silence) to 0.15 (loud speech).
# We map RMS to a 1-10 scale with class-aware bias:
#   calm -> lower arousal band (1-4)
#   mild_anxiety -> mid band (4-7)
#   high_anxiety -> high band (7-10)
AROUSAL_BANDS = {
    "calm": (1.0, 4.0),
    "mild_anxiety": (4.0, 7.0),
    "high_anxiety": (7.0, 10.0),
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rms_to_arousal(rms_val: float, label: str) -> float:
    """Map raw RMS energy into an arousal score within the class band."""
    lo, hi = AROUSAL_BANDS[label]
    # Normalize RMS into 0-1 range (clipping extreme values)
    rms_norm = min(max(rms_val / 0.10, 0.0), 1.0)
    return lo + rms_norm * (hi - lo)


def load_data():
    if not os.path.exists(EMBEDS_FILE):
        print(f"Error: {EMBEDS_FILE} not found. Run build_whisper_embeddings.py first.")
        return None, None, None

    data = np.load(EMBEDS_FILE)
    X = data['X']
    labels = data['y']

    # Check if real RMS data exists
    if 'rms' in data:
        rms_arr = data['rms']
        y_arousal = np.array(
            [rms_to_arousal(rms_arr[i], labels[i]) for i in range(len(labels))],
            dtype=np.float32,
        )
        print(f"[OK] Using real RMS-derived arousal targets (mean={y_arousal.mean():.2f})")
    else:
        # Fallback: class-center defaults
        defaults = {"calm": 2.5, "mild_anxiety": 6.0, "high_anxiety": 8.5}
        y_arousal = np.array(
            [defaults[l] for l in labels],
            dtype=np.float32,
        )
        print("[WARN] No RMS data found. Using deterministic class-center arousal fallback.")

    y_class = np.array([LABEL_MAP[l] for l in labels], dtype=np.int64)
    if X.shape[0] != y_class.shape[0] or X.shape[0] != y_arousal.shape[0]:
        raise ValueError(
            f"Shape mismatch: X={X.shape}, y_class={y_class.shape}, y_arousal={y_arousal.shape}"
        )
    return X, y_class, y_arousal


# ---------------------------------------------------------------
# Data Augmentation — Embedding-Space Mixup
# ---------------------------------------------------------------

def mixup_batch(x: torch.Tensor, y_class: torch.Tensor, y_arousal: torch.Tensor, alpha: float = 0.2):
    """
    Mixup augmentation in embedding space.
    
    Creates virtual training examples by linearly interpolating between
    pairs of embeddings and their labels. This regularizes the model
    and improves generalization, especially with small datasets.
    """
    if alpha <= 0:
        return x, y_class, y_arousal, 1.0
    
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 so the primary example dominates
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    # For arousal (regression), we can directly interpolate
    mixed_arousal = lam * y_arousal + (1 - lam) * y_arousal[index]
    
    return mixed_x, y_class, mixed_arousal, lam, index


# ---------------------------------------------------------------
# Label Smoothing Cross Entropy
# ---------------------------------------------------------------

class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy with label smoothing.
    
    Prevents the model from becoming overconfident on the training set
    by redistributing a small fraction of the probability mass from the
    ground-truth class to all other classes. This is especially important
    for emotion recognition where class boundaries are inherently fuzzy.
    """
    def __init__(self, num_classes: int = 3, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # One-hot encode targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Apply class weights if provided
        if self.weight is not None:
            per_sample_weight = self.weight[targets]
            loss = -(smooth_targets * log_probs).sum(dim=-1) * per_sample_weight
        else:
            loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        return loss.mean()


# ---------------------------------------------------------------
# Phase A: Train Level 2 Classifier
# ---------------------------------------------------------------

def split_dataset(X, y_class, y_arousal, val_ratio=0.15, test_ratio=0.15):
    test_plus_val = val_ratio + test_ratio
    X_train, X_tmp, yc_train, yc_tmp, ya_train, ya_tmp = train_test_split(
        X,
        y_class,
        y_arousal,
        test_size=test_plus_val,
        random_state=SEED,
        stratify=y_class,
    )

    val_fraction = val_ratio / test_plus_val
    X_val, X_test, yc_val, yc_test, ya_val, ya_test = train_test_split(
        X_tmp,
        yc_tmp,
        ya_tmp,
        test_size=(1.0 - val_fraction),
        random_state=SEED,
        stratify=yc_tmp,
    )
    return X_train, X_val, X_test, yc_train, yc_val, yc_test, ya_train, ya_val, ya_test


def train_phase_a(X_train, yc_train, ya_train, X_val, yc_val, ya_val, X_test, yc_test, ya_test):
    print("\n" + "=" * 60)
    print(" PHASE A: Training Level 2 Classifier (with Validation) ")
    print("=" * 60)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(yc_train), torch.tensor(ya_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(yc_val), torch.tensor(ya_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(yc_test), torch.tensor(ya_test))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = Level2Classifier(input_dim=512)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    # Compute class weights to handle imbalanced dataset
    class_counts = np.bincount(yc_train, minlength=3).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f" Class distribution: {dict(zip(ID2LABEL.values(), class_counts.astype(int)))}")
    print(f" Class weights: {class_weights}")

    # Label smoothing cross entropy for better generalization
    ce_loss = LabelSmoothingCE(
        num_classes=3,
        smoothing=0.1,
        weight=torch.tensor(class_weights).to(DEVICE),
    )
    mse_loss = nn.MSELoss()

    # Cosine annealing LR scheduler for smoother convergence
    epochs = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_metric = -1.0
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        for x_b, c_b, a_b in train_loader:
            x_b = x_b.to(DEVICE)
            c_b = c_b.to(DEVICE)
            a_b = a_b.unsqueeze(1).to(DEVICE)

            # Apply mixup augmentation
            mixed_x, _, mixed_a, lam, idx = mixup_batch(x_b, c_b, a_b, alpha=0.2)

            optimizer.zero_grad()
            logits, arousal_pred = model(mixed_x)

            # For mixup: blend the classification losses
            loss_c = lam * ce_loss(logits, c_b) + (1 - lam) * ce_loss(logits, c_b[idx])
            loss_a = mse_loss(arousal_pred, mixed_a)
            # Increased arousal weight from 0.1 to 0.3 for better multi-task balance
            loss = loss_c + (loss_a * 0.3)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_b, c_b, a_b in val_loader:
                x_b = x_b.to(DEVICE)
                logits, _ = model(x_b)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(c_b.numpy())

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        print(
            f" Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} "
            f"| Val Acc: {acc:.4f} | Val Macro-F1: {macro_f1:.4f} | LR: {current_lr:.2e}"
        )

        # Early stopping on macro-F1
        if macro_f1 > best_metric:
            best_metric = macro_f1
            patience_counter = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "level2_mlp.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stopping at epoch {epoch+1} (best val macro-F1={best_metric:.4f})")
                break

    # Final report on best model
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "level2_mlp.pth"), map_location=DEVICE))
    model.eval()
    all_preds, all_labels = [], []
    all_arousal_preds, all_arousal_labels = [], []
    with torch.no_grad():
        for x_b, c_b, a_b in test_loader:
            x_b = x_b.to(DEVICE)
            logits, arousal_pred = model(x_b)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(c_b.numpy())
            all_arousal_preds.extend(arousal_pred.squeeze().cpu().numpy())
            all_arousal_labels.extend(a_b.numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    final_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    # Arousal regression metrics
    arousal_preds_arr = np.array(all_arousal_preds)
    arousal_labels_arr = np.array(all_arousal_labels)
    arousal_mae = np.mean(np.abs(arousal_preds_arr - arousal_labels_arr))
    
    print(f"\n Held-out Test Accuracy: {final_acc:.4f}")
    print(f" Held-out Test Macro-F1: {final_macro_f1:.4f}")
    print(f" Arousal MAE: {arousal_mae:.3f} (on 1-10 scale)")
    print(classification_report(
        all_labels, all_preds,
        target_names=list(LABEL_MAP.keys()), zero_division=0
    ))
    print(" -> Level 2 Model saved to disk.")
    return model


# ---------------------------------------------------------------
# Phase B: Train Level 4 Temporal Bi-LSTM
# ---------------------------------------------------------------

def train_phase_b():
    print("\n" + "=" * 60)
    print(" PHASE B: Training Level 4 Temporal Bi-LSTM + Attention ")
    print("=" * 60)

    # Synthetic escalation sequences (we don't have real 30-sec continuous data)
    n_samples = 2000  # Doubled from 1000 for better generalization
    rng = np.random.default_rng(SEED)

    # Escalating patterns (target = 1): monotonically increasing with noise
    escalating = np.zeros((n_samples, 6, 1), dtype=np.float32)
    for i in range(n_samples):
        base = np.sort(rng.uniform(3, 9, 6))
        escalating[i, :, 0] = base + rng.normal(0, 0.15, 6)

    # Non-escalating patterns (target = 0): flat, random, or decreasing
    non_escalating = np.zeros((n_samples, 6, 1), dtype=np.float32)
    for i in range(n_samples):
        pattern_type = rng.choice(['flat', 'random', 'decreasing', 'spike_return'])
        if pattern_type == 'flat':
            base_val = rng.uniform(2, 8)
            non_escalating[i, :, 0] = base_val + rng.normal(0, 0.5, 6)
        elif pattern_type == 'random':
            non_escalating[i, :, 0] = rng.uniform(2, 9, 6)
        elif pattern_type == 'spike_return':
            # Spike then return to baseline (false positive pattern)
            baseline = rng.uniform(3, 5)
            seq = [baseline, baseline + rng.uniform(2, 4), baseline + rng.uniform(0, 1),
                   baseline, baseline + rng.uniform(-1, 1), baseline]
            non_escalating[i, :, 0] = np.array(seq)
        else:
            base = np.sort(rng.uniform(3, 9, 6))[::-1]
            non_escalating[i, :, 0] = base + rng.normal(0, 0.15, 6)

    X_seq = np.vstack([escalating, non_escalating])
    y_seq = np.vstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))]).astype(np.float32)

    # Shuffle
    idx = rng.permutation(len(X_seq))
    X_seq, y_seq = X_seq[idx], y_seq[idx]

    # Split
    split = int(0.8 * len(X_seq))
    train_ds = TensorDataset(torch.tensor(X_seq[:split]), torch.tensor(y_seq[:split]))
    test_ds = TensorDataset(torch.tensor(X_seq[split:]), torch.tensor(y_seq[split:]))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    lstm_net = TemporalContextLSTM()
    lstm_net.to(DEVICE)

    optimizer = optim.AdamW(lstm_net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    bce_loss = nn.BCELoss()

    best_acc = 0.0
    epochs = 15
    for epoch in range(epochs):
        lstm_net.train()
        total_loss = 0
        for x_seq, labels in train_loader:
            x_seq, labels = x_seq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = lstm_net(x_seq)
            loss = bce_loss(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        lstm_net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_seq, labels in test_loader:
                x_seq, labels = x_seq.to(DEVICE), labels.to(DEVICE)
                preds = lstm_net(x_seq)
                predicted = (preds > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f" Epoch {epoch+1:02d}/{epochs} | BCE Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(lstm_net.state_dict(), os.path.join(MODELS_DIR, "level4_bilstm.pth"))

    print(f" Best Val Acc: {best_acc:.4f}")
    print(" -> Level 4 Bi-LSTM + Attention saved to disk.")


# ---------------------------------------------------------------
# Execute
# ---------------------------------------------------------------

def execute_pipeline(skip_level4: bool = False):
    set_seed(SEED)
    X, y_class, y_arousal = load_data()
    if X is None:
        return

    print(f"\n Dataset size: {len(X)} samples")
    print(f" Label distribution: calm={sum(y_class==0)}, mild={sum(y_class==1)}, high={sum(y_class==2)}")

    X_train, X_val, X_test, yc_train, yc_val, yc_test, ya_train, ya_val, ya_test = split_dataset(
        X, y_class, y_arousal
    )
    print(
        f" Split sizes: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
    )

    train_phase_a(X_train, yc_train, ya_train, X_val, yc_val, ya_val, X_test, yc_test, ya_test)
    if skip_level4:
        print("[WARN] Skipping Level 4 synthetic Bi-LSTM training by request.")
    else:
        train_phase_b()

    print("\n" + "*" * 60)
    print(" FULL TRAINING PIPELINE COMPLETE! ")
    print("*" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AuraOS 4-level architecture.")
    parser.add_argument(
        "--skip-level4",
        action="store_true",
        help="Skip synthetic Level 4 Bi-LSTM training pass.",
    )
    args = parser.parse_args()
    execute_pipeline(skip_level4=args.skip_level4)
