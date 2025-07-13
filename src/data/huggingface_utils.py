from huggingface_hub import hf_hub_download
import os
import shutil

def download_huggingface_files(repo_id, hf_dir, hf_filenames, local_dir):

    # Make sure destination exists
    os.makedirs(local_dir, exist_ok=True)

    # Download each file
    for filename in hf_filenames:
        file_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{local_dir}/{filename}",
        )
        shutil.copy(file_path, os.path.join(local_dir, filename))
