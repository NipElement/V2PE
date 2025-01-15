#!/bin/bash
export MASTER_ADDR=localhost
export WORLD_SIZE=16  # Total number of processes
GPUS_PER_NODE=8
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

BATCH_SIZE=1
TASK_COUNT=${#tasks[@]}
BASE_PORT=15432


# Loop through each task and run evaluation
for ((i=0; i<TASK_COUNT; i+=BATCH_SIZE)); do
    echo "$(date) Starting batch $((i/BATCH_SIZE + 1))"

    # Run a batch of tasks
    for ((j=0; j<BATCH_SIZE && i+j<TASK_COUNT; j++)); do
        task="${tasks[$((i+j))]}"
        MASTER_PORT=$((BASE_PORT + i + j))  # Unique port for each task

        echo "$(date) Starting task: ${task} on port ${MASTER_PORT}"

        torchrun \
            --nnodes=2 \
            --nproc_per_node=$GPUS_PER_NODE  \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            eval/mm_niah/eval_mm_niah.py \
            --checkpoint $CHECKPOINT \
            --outputs-dir $LOG_DIR \
            --task $task \
            --num-gpus-per-rank 1 \
            > "${LOG_DIR}/${task}.log" 2>&1 &
    done

    sleep 0.2 # Wait for the current batch to finish
    wait

    echo "$(date) Finished batch $((i/BATCH_SIZE + 1))"
done

echo "All tasks completed!"