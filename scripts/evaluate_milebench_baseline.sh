#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

NUM_GPUS_PER_NODE=8

CHECKPOINT="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage1_mammoth_baseline"
BASE_LOG_DIR="eval_logs/milebench"

model_name="internvl2_5_8b_stage1_mammoth-baseline"

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

declare -a strides=(8 16 32 128 256)

for stride in "${strides[@]}"; do
    LOG_DIR="${BASE_LOG_DIR}/${model_name}-v2pe-stride${stride}"
    mkdir -p "$LOG_DIR"
    
    for task in "${tasks[@]}"; do
        echo "$(date) Running ${model_name}_${task} with rope_pos_id_stride=${stride}"
        
        torchrun \
            --nproc_per_node "$NUM_GPUS_PER_NODE" \
            --standalone \
            eval/milebench/eval_milebench.py \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$LOG_DIR" \
            --dataset_name "$task" \
            --num-gpus-per-rank 1 \
            --rope_pos_id_version v2pe_fix \
            --rope_pos_id_stride "$stride" \
            > "${LOG_DIR}/${task}.log" 2>&1

        sleep 0.2
    done
done

echo "All tasks completed!"
