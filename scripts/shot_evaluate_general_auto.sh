set -x
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi
NUM_GPUS_PER_NODE=8
# CUDA_VISIBLE_DEVICES=0
CHECKPOINT="/map-vepfs/yuansheng/LongContext/trained_models/invervl2_5_8b_stage1_mammoth"
# CHECKPOINT="/map-vepfs/yuansheng/LongContext/V2PE/pretrained/InternVL2_5-8B"

declare -a tasks=( \
    # 'vqa-chartqa-test' \
    # 'vqa-docvqa-val' \
    # 'vqa-ai2d-test' \
    # 'vqa-infovqa-val' \
    # 'scienceqa' \
    # 'pope' \
    'mmmu-val' \
    # 'mmbench-test-en' \
    # 'seed' \
)

declare -a strides=(64)
# declare -a shots=(1 2 3 5 10 20)
declare -a shots=(1)
for shot in "${shots[@]}"; do
    for stride in "${strides[@]}"; do
        for ((j=0; j<${#tasks[@]}; j++)); do
            model_path=$CHECKPOINT
            task=${tasks[j]}

            model_name="$(basename ${model_path})"
            model_name="test"
            LOG_DIR="/map-vepfs/yuansheng/LongContext/V2PE/eval_logs/${task}_${shot}/${model_name}-v2pe-stride${stride}"
            mkdir -p "$LOG_DIR"
            echo "$(date) ${model_name}_${task}"

            if [ "${task}" == "vqa-chartqa-test" ]; then
                srun \
                    -p ${PARTITION} \
                    --gres=gpu:8 \
                    --ntasks=1 \
                    --quotatype=${QUOTA_TYPE} \
                    --ntasks-per-node=1 \
                    -o "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
                    -e "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
                    --async \
                sh scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 12 "${ARGS[@]:1}"
            elif [ "${task}" == "vqa-infovqa-val" ]; then
                srun \
                    -p ${PARTITION} \
                    --gres=gpu:8 \
                    --ntasks=1 \
                    --quotatype=${QUOTA_TYPE} \
                    --ntasks-per-node=1 \
                    -o "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
                    -e "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
                    --async \
                sh scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 24 "${ARGS[@]:1}"
            elif [ "${task}" == "vqa-docvqa-val" ]; then
                srun \
                    -p ${PARTITION} \
                    --gres=gpu:8 \
                    --ntasks=1 \
                    --quotatype=${QUOTA_TYPE} \
                    --ntasks-per-node=1 \
                    -o "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
                    -e "$CHECKPOINT/eval_origin_$STRIDE/${task}.log" \
                    --async \
                sh scripts/evaluate.sh ${model_path} ${task} --dynamic --max-num 18 "${ARGS[@]:1}"
            elif [ "${task}" == "scienceqa" ]; then
                torchrun \
                    --nproc_per_node $NUM_GPUS_PER_NODE \
                    --standalone \
                    eval/scienceqa/evaluate_scienceqa.py \
                    --checkpoint ${CHECKPOINT} \
                    --datasets sqa_test \
                    --out-dir ${LOG_DIR} \
                    --rope_pos_id_version v2pe_fix \
                    --rope_pos_id_stride 64
                    # > "${LOG_DIR}/${task}_${model_name}.log" 2>&1
            elif [ "${task}" == "mmmu-val" ]; then
                torchrun \
                    --nproc_per_node $NUM_GPUS_PER_NODE \
                    --standalone \
                    eval/mmmu/shot_evaluate_mmmu.py \
                    --checkpoint ${CHECKPOINT} \
                    --datasets MMMU_validation \
                    --out-dir ${LOG_DIR} \
                    --rope_pos_id_version v2pe_fix \
                    --rope_pos_id_stride "$stride" \
                    --n_shot "$shot" \
                    --example-max-patch 6 \
                    # > "${LOG_DIR}/${task}_${model_name}.log" 2>&1
            fi
        done
    done
done