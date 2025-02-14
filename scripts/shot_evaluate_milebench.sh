#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi
NUM_GPUS_PER_NODE=8
# export RANK=$MLP_ROLE_INDEX
# export WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_WORKER_GPU))
# export MASTER_ADDR=$MLP_WORKER_0_HOST
# export MASTER_PORT=$MLP_WORKER_0_PORT
# export LOCAL_RANK=$(($RANK % $MLP_WORKER_GPU))

CHECKPOINT="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage1_mammoth/checkpoint-4375"

model_name="internvl2_5_8b_stage1_mammoth"

declare -a strides=(64)
# declare -a strides=(1 2 4 8 16 32 128 256)
declare -a shots=(1 2 3 5 10 20)
# Task list
declare -a tasks=( \
    # 'ALFRED' \
    # 'ActionLocalization' \
    # 'ActionPrediction' \
    # 'ActionSequence' \
    # 'CLEVR-Change' \
    # 'CharacterOrder' \
    # 'CounterfactualInference' \
    # 'DocVQA' \
    # 'EgocentricNavigation' \
    # 'GPR1200' \
    # 'IEdit' \
    # 'ImageNeedleInAHaystack' \
    # 'MMCoQA' \
    # 'MovingAttribute' \
    # 'MovingDirection' \
    # 'MultiModalQA' \
    # 'OCR-VQA' \
    # 'ObjectExistence' \
    # 'ObjectInteraction' \
    # 'ObjectShuffle' \
    # 'SceneTransition' \
    'SlideVQA' \
    # 'Spot-the-Diff' \
    # 'StateChange' \
    # 'TQA' \
    # 'TextNeedleInAHaystack' \
    # 'WebQA' \
    # 'WikiVQA' \
    # 'nuscenes' \
)
for shot in "${shots[@]}"; do
    for stride in "${strides[@]}"; do
        LOG_DIR="eval_logs/milebench_${shot}shot/${model_name}-v2pe-stride${stride}"
        mkdir -p "$LOG_DIR"
        
        for task in "${tasks[@]}"; do
            echo "$(date) Running ${model_name}_${task} with rope_pos_id_stride=${stride} and n-shot=${shot}"
            torchrun \
                --nproc_per_node "$NUM_GPUS_PER_NODE" \
                --standalone \
                eval/milebench/eval_milebench_shot.py \
                --checkpoint "$CHECKPOINT" \
                --output_dir "$LOG_DIR" \
                --dataset_name "$task" \
                --rope_pos_id_version v2pe_fix \
                --rope_pos_id_stride "$stride" \
                --num-gpus-per-rank 1 \
                --n-shot "$shot" \
                --example-seed 42 \
                --example-max-patch 6 \
                > "${LOG_DIR}/${task}.log" 2>&1
            sleep 0.2
        done
    done
done
