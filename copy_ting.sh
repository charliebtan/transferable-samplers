#!/bin/bash

ALL_DATES=(
                "2025-05-10_02-21-17"
                "2025-05-11_18-51-26"
                "2025-05-11_18-49-32"
                "2025-05-11_18-55-20"
                "2025-05-11_18-55-02"
                "2025-05-11_18-52-16"
                "2025-05-11_01-44-04"
                "2025-05-11_18-55-55"
                "2025-05-11_18-49-08"
                "2025-05-11_18-55-39"
                "2025-05-11_18-49-54"
                "2025-05-13_20-21-00"
                "2025-05-13_20-19-58"
)

for DATE in "${ALL_DATES[@]}"; do
    SRC="/network/archive/t/tanc/self-consume-bg-logs/eval/multiruns/$DATE/*"
    DEST="../scratch/ecnf_samples/"
    echo "Copying from $SRC to $DEST"
    cp -r $SRC $DEST
done