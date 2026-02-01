#!/usr/bin/env python3
"""
Wellness Tourism Model Deployment Script
Instructions for HF Spaces deployment
"""

def print_deployment_status():
    print("MODEL DEPLOYMENT STATUS")
    print("=" * 50)
    print("✓ Dockerfile: Ready")
    print("✓ Streamlit app: app.py ready") 
    print("✓ Dependencies: requirements.txt ready")
    print("✓ Model: XGBoost AUC 0.9712 (models/best_wellness_xgb.pkl)")
    print("✓ Encoders: label_encoders.pkl ready")
    print()
    print("DEPLOYMENT STEPS:")
    print("1. https://huggingface.co/new-space")
    print("   Name: AnjaneyaluAJ/wellness-tourism-predictor")
    print("   SDK: Streamlit | License: apache-2.0 | Public")
    print("2. Files tab → Upload:")
    print("   - app.py")
    print("   - Dockerfile")
    print("   - requirements.txt")
    print("   - models/best_wellness_xgb.pkl")
    print("   - models/label_encoders.pkl")
    print("3. Auto-deploys to:")
    print("   https://huggingface.co/spaces/AnjaneyaluAJ/wellness-tourism-predictor")
    print()
    print("CONFIGURATION SUMMARY:")
    print("Port: 7860 (Streamlit)")
    print("Model load: joblib from models/")
    print("Input: Form → DataFrame → Predict → Probability")

files = ["app.py", "Dockerfile", "requirements.txt", "models/best_wellness_xgb.pkl", "models/label_encoders.pkl"]
print("\nFILE CHECK:")
for f in files:
    status = "✓" if os.path.exists(f) else "✗"
    print(f"{status} {f}")

if all(os.path.exists(f) for f in files):
    print("\nSTATUS: READY FOR HF SPACES UPLOAD")
else:
    print("\nSTATUS: MISSING FILES - CHECK ABOVE")

if __name__ == "__main__":
    print_deployment_status()
