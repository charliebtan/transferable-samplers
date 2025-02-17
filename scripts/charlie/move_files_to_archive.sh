#!/bin/bash

# Define old and new prefix
OLD_PREFIX="/home/mila/b/bosejoey/scratch/"
NEW_PREFIX="/network/archive/b/bosejoey/"

# List of files to transfer
FILES=(
    # MAIN MODELS
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/12/checkpoints/epoch_699_cropped.ckpt" # AL2
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/4/checkpoints/epoch_899_cropped.ckpt" # AL2
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-24_07-35-01/0/checkpoints/epoch_699_cropped.ckpt" # AL2
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/2/checkpoints/epoch_899_cropped.ckpt" # AL3
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/0/checkpoints/epoch_949_cropped.ckpt" # AL3
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/1/checkpoints/epoch_999_cropped.ckpt" # AL3
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/0/checkpoints/epoch_749_cropped.ckpt" # AL4
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/1/checkpoints/epoch_999_cropped.ckpt" # AL4
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/2/checkpoints/epoch_999_cropped.ckpt" # AL4
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/runs/tarflow_al6_v2/checkpoints/epoch_999_time.ckpt" # AL6
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/runs/chignolin_tarflow_moonshot/checkpoints/epoch_799_time.ckpt" # DECA
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/runs/chignolin_tarflow_moonshot/checkpoints/epoch_999_time.ckpt" # DECA

    # ABLATION MODELS
)

# Loop through each file in FILES
for SRC in "${FILES[@]}"; do
    # Replace OLD_PREFIX with NEW_PREFIX in the destination path
    DEST=$(echo "$SRC" | sed "s|$OLD_PREFIX|$NEW_PREFIX|")

    echo "Copying $SRC to $DEST"

    # Ensure destination directory exists
    mkdir -p "$(dirname "$DEST")"

    # Copy the file
    cp "$SRC" "$DEST"
done

