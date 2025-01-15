#!/bin/bash
export MASTER_ADDR=localhost
export WORLD_SIZE=8  # Total number of processes
GPUS=$WORLD_SIZE
CHECKPOINT="pretrained/InternVL2_5-8B"
LOG_DIR=eval_logs/mmniah_long/internvl2_5_8b

mkdir -p $LOG_DIR  # Create log directory


declare -a tasks=( \

    'retrieval-image-test-128k' \
    'retrieval-image-test-256k' \
    'retrieval-image-test-512k' \
    'retrieval-image-test-1M' \
)

model_name="internvl2_5"

BATCH_SIZE=1
TASK_COUNT=${#tasks[@]}
# Loop through each task and run evaluation
for ((i=0; i<TASK_COUNT; i+=BATCH_SIZE)); do
    echo "$(date) Starting batch $((i/BATCH_SIZE + 1))"

    # Run a batch of tasks
    for ((j=0; j<BATCH_SIZE && i+j<TASK_COUNT; j++)); do
        task="${tasks[$((i+j))]}"
        MASTER_PORT=$((15432 + i + j))  # Unique port for each task

        echo "$(date) Starting task: ${task} on port ${MASTER_PORT}"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        torchrun \
            --nproc_per_node=$GPUS \
            --master_port=$MASTER_PORT \
            eval/mm_niah/eval_mm_niah.py \
            --checkpoint $CHECKPOINT \
            --outputs-dir $LOG_DIR \\
            --ring_attn \
            --task $task \
            --num-gpus-per-rank 1 \
            > "${LOG_DIR}/${task}.log" 2>&1 &
    done

    sleep 0.2 # Wait for the current batch to finish
    wait

    echo "$(date) Finished batch $((i/BATCH_SIZE + 1))"
done

echo "All tasks completed!"