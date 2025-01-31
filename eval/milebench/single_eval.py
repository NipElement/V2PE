import os
import json
import argparse
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/map-vepfs/yuansheng/LongContext/V2PE/dataset/benchmark/MileBench', help='Path to MileBench dataset directory')
    parser.add_argument('--output_root', required=True, help='Path to directory containing model output folders')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of parallel threads')  # 线程数参数

    args = parser.parse_args()
    return args

def get_all_datasets(data_dir):
    """ 遍历 `data_dir` 下所有子数据集，排除非数据集文件夹 """
    excluded_dirs = {".cache", "preview", "images", "__pycache__"} 
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d not in excluded_dirs]

def get_all_model_dirs(output_root):
    """ 遍历 `output_root` 目录下所有模型文件夹 """
    excluded_dirs = {"__pycache__"}  # 可以扩展
    return [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d)) and d not in excluded_dirs]

def run_evaluation(data_dir, dataset_name, model_output_dir, overwrite):
    """ 运行 `evaluate.py` 进行评估 """
    result_dir = os.path.join(model_output_dir, dataset_name)
    output_pth = os.path.join(result_dir, "pred.json")

    if not os.path.exists(output_pth):
        return f"❌ Error: {output_pth} does not exist. Skipping {dataset_name}."

    if overwrite:
        # **删除旧的评估结果**
        for filename in ["eval.json", "eval_score.json", "pred_with_extracted.json"]:
            file_path = os.path.join(result_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    # **运行评估**
    cmd_string = f'python /map-vepfs/yuansheng/LongContext/V2PE/eval/milebench/evaluate.py  \
                    --data-dir {data_dir} \
                    --dataset {dataset_name} \
                    --result-dir {result_dir}'

    try:
        subprocess.run(cmd_string, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return f"❌ Evaluation failed for {dataset_name}: {e}"

    return None  # 成功返回 None

def process_model(model_output_dir, data_dir, dataset_names, overwrite):
    """ 处理一个模型的所有数据集 """
    error_messages = []

    # **删除整个模型的 milebench_result.json 和 milebench_result.csv**
    if overwrite:
        for filename in ["milebench_result.json", "milebench_result.csv"]:
            file_path = os.path.join(model_output_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    # **并行运行所有数据集的评估**
    with ThreadPoolExecutor(max_workers=4) as executor:  # 4 线程并行
        future_to_dataset = {executor.submit(run_evaluation, data_dir, dataset, model_output_dir, overwrite): dataset for dataset in dataset_names}

        for future in tqdm(as_completed(future_to_dataset), total=len(dataset_names), desc=f"Evaluating {os.path.basename(model_output_dir)}", unit="dataset"):
            error = future.result()
            if error:
                error_messages.append(error)

    return error_messages

def main(args):
    dataset_names = get_all_datasets(args.data_dir)
    model_dirs = get_all_model_dirs(args.output_root)
    all_errors = []

    # **并行处理多个模型**
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_model = {executor.submit(process_model, os.path.join(args.output_root, model_dir), args.data_dir, dataset_names, args.overwrite): model_dir for model_dir in model_dirs}

        for future in tqdm(as_completed(future_to_model), total=len(model_dirs), desc="Processing Model Output", unit="model"):
            model_errors = future.result()
            if model_errors:
                all_errors.extend(model_errors)

    # **只打印错误信息**
    if all_errors:
        print("\n🔴 Errors encountered during evaluation:")
        for error in all_errors:
            print(error)

    print(f"\n🎯 All evaluations completed!")

if __name__ == '__main__':
    args = parse_args()
    main(args)
'''
python eval/milebench/single_eval.py \
    --output_root /map-vepfs/yuansheng/LongContext/V2PE/eval/milebench \
    --overwrite \
    --num_threads 16
'''