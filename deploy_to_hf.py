from huggingface_hub import HfApi, login
import os

# Configuration
SPACE_NAME = "AnjaneyaluAJ/wellness-tourism-predictor"
MODEL_REPO = "AnjaneyaluAJ/wellness-tourism-model"

# HF Token (set as env var or manual)
HF_TOKEN = "hf_your_write_token"  # Replace or use os.getenv('HF_TOKEN')

def deploy():
    login(HF_TOKEN)
    api = HfApi()
    
    # Create Space if not exists
    try:
        api.create_repo(SPACE_NAME, repo_type="space", space_sdk="streamlit")
        print(f"Space created: https://huggingface.co/spaces/{SPACE_NAME}")
    except:
        print("Space exists")
    
    # Push deployment files
    files_to_upload = [
        "app.py",
        "requirements.txt", 
        "Dockerfile",
        "models/best_wellness_xgb.pkl",
        "models/label_encoders.pkl"
    ]
    
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=SPACE_NAME,
                repo_type="space"
            )
            print(f"Uploaded: {file_path}")
    
    print("Deployment complete!")
    print(f"Space URL: https://huggingface.co/spaces/{SPACE_NAME}")

if __name__ == "__main__":
    deploy()

