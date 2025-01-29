import wandb
import os

# Function to download media from a specific run
def download_wandb_media(project_name, run_id, media_type, output_dir):
    """
    Downloads media files from a specific Weights & Biases run.

    Parameters:
        project_name (str): The name of the W&B project (e.g., "my-project").
        run_id (str): The ID of the specific W&B run.
        media_type (str): The type of media to download (e.g., "images", "videos").
        output_dir (str): Directory to save the downloaded media files.
    """
    # Authenticate with W&B
    wandb.login()

    # Initialize API
    api = wandb.Api()

    # Get the run
    run = api.run(f"{project_name}/{run_id}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading {media_type} from run {run_id}...")

    for file in run.files():
        file.download(root=output_dir)

    print("Download complete.")

# Example usage
if __name__ == "__main__":
    # Replace these variables with your specific project details
    PROJECT_NAME = "openproblems-comp/fast-tbg"

    run_ids = ["a2sza44y", "88dkov34", "n9abvbac"]

    names = ["al2_tarflow", "al3_tarflow", "al4_tarflow"]

    for run_id, name in zip(run_ids, names):
        OUTPUT_DIR = f"./{name}"
        download_wandb_media(PROJECT_NAME, run_id, "images", OUTPUT_DIR)