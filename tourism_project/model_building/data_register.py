from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Hugo014/Tourism-Package-Prediction"
repo_type = "dataset"

# Set token 
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

# Check and create repo
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating repository '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.environ["HF_TOKEN"])
    print(f"Repository created!")

# Upload the local data folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Upload complete!")
