#!/usr/bin/env python3
"""
Wellness Tourism Model - Manual HF Spaces Deployment Instructions
Run: python deploy_to_hf.py (prints steps + checks files)
"""

REQUIRED_FILES = [
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "models/best_wellness_xgb.pkl",
    "models/label_encoders.pkl"
]

def check_files():
    print("HF SPACES DEPLOYMENT CHECK")
    print("=" * 50)
    all_present = True
    for f in REQUIRED_FILES:
        if os.path.exists(f):
            print(f"✓ {f}")
        else:
            print(f"✗ Missing: {f}")
            all_present = False
    print("\nSTEPS:")
    print("1. https://huggingface.co/new-space")
    print("2. Owner: AnjaneyaluAJ | Name: wellness-tourism-predictor")
    print("3. SDK: Gradio | Visibility: Public | Create")
    print("4. Files tab → Upload all ✓ files above")
    print("5. Auto-deploys: https://huggingface.co/spaces/AnjaneyaluAJ/wellness-tourism-predictor")
    return "READY" if all_present else "Fix missing files first"

if __name__ == "__main__":
    print(check_files())
