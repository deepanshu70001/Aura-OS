import argparse
import os
import sys


REQUIRED_MODELS = (
    "level2_mlp.pth",
    "level4_bilstm.pth",
)


def _models_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "models"))


def verify_models(strict: bool = False) -> int:
    models_dir = _models_dir()
    os.makedirs(models_dir, exist_ok=True)

    print(f"Checking model directory: {models_dir}")
    missing = []
    for model_name in REQUIRED_MODELS:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            missing.append(model_name)
            print(f"[MISSING] {model_name}")
            continue
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[OK] {model_name} ({size_mb:.2f} MB)")

    if not missing:
        print("\nAll required Level-2 stack models are available.")
        return 0

    print("\nSome required models are missing.")
    print("Train and export models with:")
    print("  1) python training/build_whisper_embeddings.py")
    print("  2) python training/train_new_architecture.py")
    print("The backend can still run, but audio inference will partially fallback.")

    return 1 if strict else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify required AuraOS model artifacts.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any required model is missing.",
    )
    args = parser.parse_args()
    sys.exit(verify_models(strict=args.strict))
