from huggingface_hub import HfApi, create_repo
import os

# Initialize the API
api = HfApi()

# Configuration
local_model_path = "/scratch/ap9111/ReinforcedTeacher/outputs/qwen3b_rlt_grpo"  # Change this to your local path
repo_id = "ArunP3799/qwen3b_rlt_grpo"  # Change to your desired repo name
repo_type = "model"  # Can be "model", "dataset", or "space"

# Optional: Create the repository if it doesn't exist
try:
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,  # Set to True if you want a private repo
        exist_ok=True
    )
    print(f"Repository {repo_id} created/verified!")
except Exception as e:
    print(f"Error creating repo: {e}")

# Upload all files from the local folder
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print(f"Model successfully pushed to https://huggingface.co/{repo_id}")