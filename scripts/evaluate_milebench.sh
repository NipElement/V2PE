#!/bin/bash
export MASTER_ADDR=localhost
export WORLD_SIZE=7  # Total number of processes
GPUS=$WORLD_SIZE
CHECKPOINT="pretrained/InternVL2_5-8B"
LOG_DIR=eval_logs/milebench/internvl2_5_8b

mkdir -p $LOG_DIR  # Create log directory

# Task list
declare -a tasks=( \
    'ALFRED' \
    'ActionLocalization' \
    'ActionPrediction' \
    'ActionSequence' \
    'CLEVR-Change' \
    'CharacterOrder' \
    'CounterfactualInference' \
    'DocVQA' \
    'EgocentricNavigation' \
    'GPR1200' \
    'IEdit' \
    'ImageNeedleInAHaystack' \
    'MMCoQA' \
    'MovingAttribute' \
    'MovingDirection' \
    'MultiModalQA' \
    'OCR-VQA' \
    'ObjectExistence' \
    'ObjectInteraction' \
    'ObjectShuffle' \
    'SceneTransition' \
    'SlideVQA' \
    'Spot-the-Diff' \
    'StateChange' \
    'TQA' \
    'TextNeedleInAHaystack' \
    'WebQA' \
    'WikiVQA' \
    'nuscenes' \
)

model_name="internvl2_5"

BATCH_SIZE=2
TASK_COUNT=${#tasks[@]}
# Loop through each task and run evaluation
for ((i=0; i<TASK_COUNT; i+=BATCH_SIZE)); do
    echo "$(date) Starting batch $((i/BATCH_SIZE + 1))"

    # Run a batch of tasks
    for ((j=0; j<BATCH_SIZE && i+j<TASK_COUNT; j++)); do
        task="${tasks[$((i+j))]}"
        MASTER_PORT=$((15432 + i + j))  # Unique port for each task

        echo "$(date) Starting task: ${task} on port ${MASTER_PORT}"

        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
        torchrun \
            --nproc_per_node=$GPUS \
            --master_port=$MASTER_PORT \
            eval/milebench/eval_milebench.py \
            --checkpoint $CHECKPOINT \
            --output_dir $LOG_DIR \
            --dataset_name $task \
            --num-gpus-per-rank 1 \
            > "${LOG_DIR}/${task}.log" 2>&1 &
    done

    sleep 0.2 # Wait for the current batch to finish
    wait

    echo "$(date) Finished batch $((i/BATCH_SIZE + 1))"
done

echo "All tasks completed!"
