#!/bin/bash
export MASTER_ADDR=localhost
CHECKPOINT="pretrained/InternVL2_5-8B"
LOG_DIR=eval_logs/mmniah/internvl2_5_8b

mkdir -p $LOG_DIR  # Create log directory

declare -a tasks=( \
    'retrieval-text-test' \
    'retrieval-image-test' \
    'counting-text-test' \
    'counting-image-test' \
    'reasoning-text-test' \
    'reasoning-image-test' \
)

model_name="internvl2_5"

# Loop through each task and run evaluation
for task in "${tasks[@]}"; do
    echo "$(date) Starting task: ${task}"

    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    torchrun \
        --nproc_per_node=1 \
        eval/mm_niah/eval_mm_niah.py \
        --checkpoint $CHECKPOINT \
        --outputs-dir $LOG_DIR \
        --task $task \
        --num-gpus-per-rank=4 \
        > "${LOG_DIR}/${task}.log" 2>&1
    wait
    echo "$(date) Finished task: ${task}"
done

echo "All tasks completed!"
