#!/bin/bash

# List of files to transfer
FILES=(
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/12/checkpoints/epoch_699_cropped.ckpt"
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-23_22-29-10/4/checkpoints/epoch_899_cropped.ckpt"
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-24_07-35-01/0/checkpoints/epoch_699_cropped.ckpt"
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/2/checkpoints/epoch_899_cropped.ckpt"
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/0/checkpoints/epoch_949_cropped.ckpt"
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-57-11/1/checkpoints/epoch_999_cropped.ckpt"
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/0/checkpoints/epoch_749_cropped.ckpt"
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/1/checkpoints/epoch_999_cropped.ckpt"
    "/home/mila/b/bosejoey/scratch/fast-tbg/logs/train/multiruns/2025-01-21_19-56-57/2/checkpoints/epoch_999_cropped.ckpt"
    "/network/scratch/b/bosejoey/fast-tbg/logs/train/runs/chignolin_tarflow_moonshot/checkpoints/epoch_799_time.ckpt"
)

DEST_FILE=(
    "/home/ubuntu/scratch/al2/0.ckpt"
    "/home/ubuntu/scratch/al2/1.ckpt"
    "/home/ubuntu/scratch/al2/2.ckpt"
    "/home/ubuntu/scratch/al3/0.ckpt"
    "/home/ubuntu/scratch/al3/1.ckpt"
    "/home/ubuntu/scratch/al3/2.ckpt"
    "/home/ubuntu/scratch/al4/0.ckpt"
    "/home/ubuntu/scratch/al4/1.ckpt"
    "/home/ubuntu/scratch/al4/2.ckpt"
    "/home/ubuntu/scratch/deca/epoch_799_time.ckpt"
)

# Source SSH config alias
SRC_HOST="mila"
# Destination SSH config alias
DST_HOST="lambda"

for FILE in "${FILES[@]}"; do

    # Get the index of the current file
    INDEX=$(echo "${FILES[@]}" | tr ' ' '\n' | grep -n "^$FILE$" | cut -d: -f1)
    INDEX=$((INDEX - 1))

    # Get the corresponding destination file
    DEST_FILE="${DEST_FILE[$INDEX]}"

    # Transfer the file
    scp "$SRC_HOST:$FILE" "$DST_HOST:$DEST_FILE"

    # Check if SCP was successful
    if [[ $? -eq 0 ]]; then
        echo "Successfully transferred: $FILE -> $DEST_FILE"
    else
        echo "Failed to transfer: $FILE"
    fi
done
