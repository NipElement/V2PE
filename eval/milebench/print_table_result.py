import json
import pandas as pd
from tabulate import tabulate  # 确保安装了 tabulate 库：pip install tabulate
import os

# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 定义 MileBench 分类
TASK_CATEGORIES = {
    "T-1": [
        'ActionLocalization', 'ActionPrediction', 'ActionSequence'
    ],
    "T-2": [
        'ObjectExistence', 'ObjectInteraction', 'MovingAttribute', 'ObjectShuffle'
    ],
    "T-3": [
        'EgocentricNavigation', 'MovingDirection'
    ],
    "T-4": [
        'CounterfactualInference', 'StateChange', 'CharacterOrder', 'SceneTransition'
    ],
    "S-1": [
        'WebQA', 'TQA', 'MultiModalQA', 'WikiVQA'
    ],
    "S-2": [
        'SlideVQA', 'OCR-VQA', 'DocVQA'
    ],
    "S-3": [
        'Spot-the-Diff', 'CLEVR-Change', 'IEdit'
    ],
    "S-4": [
        'MMCoQA', 'ALFRED'
    ],
    "S-5": [
        'nuscenes'
    ],
    "N-1": [
        'TextNeedleInAHaystack'
    ],
    "N-2": [
        'ImageNeedleInAHaystack'
    ],
    "I-1": [
        'GPR1200'
    ]
}

# 计算每个分类的平均分
def calculate_averages(data, task_categories):
    results = []
    for category, tasks in task_categories.items():
        scores = []
        for task in tasks:
            for sub_category, sub_data in data.items():
                if task in sub_data:
                    accuracy = sub_data[task].get("Accuracy", None)
                    rouge_l = sub_data[task].get("Rouge-L f", None)
                    if accuracy is not None:
                        scores.append(accuracy)
                    elif rouge_l is not None:
                        scores.append(rouge_l)
        avg_score = sum(scores) / len(scores) if scores else 0
        results.append({"Category": category, "Average Score": avg_score * 100})
    return results

# 输出结果为表格
def output_table(results, model_name):
    shortened_name = model_name.replace("internvl2_5_8b_stage1_mammoth-", "")
    df = pd.DataFrame(results)
    df[shortened_name] = df["Average Score"].map(lambda x: f"{x:.1f}")
    df = df.drop(columns=["Average Score"])
    return df

# 主函数
def main():
    # base_path = "/map-vepfs/yuansheng/LongContext/V2PE/eval_logs/milebench"
    base_path = "/map-vepfs/yuansheng/LongContext/V2PE/eval_logs/milebench_stage2"
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    all_results = []
    for model_dir in model_dirs:
        file_path = os.path.join(base_path, model_dir, "milebench_result.json")
        if os.path.exists(file_path):
            data = load_json(file_path)
            averages = calculate_averages(data, TASK_CATEGORIES)
            model_table = output_table(averages, model_dir)
            all_results.append(model_table)
        else:
            print(f"Result file not found for model: {model_dir}")

    # 合并所有模型的结果
    if all_results:
        final_table = pd.concat(all_results, axis=1)
        final_table = final_table.loc[:, ~final_table.columns.duplicated()]  # 去重列

        # 对模型列进行排序
        sorted_columns = ["Category"] + sorted([col for col in final_table.columns if col != "Category"], key=str.lower)
        final_table = final_table[sorted_columns]

        print(tabulate(final_table, headers="keys", tablefmt="plain", showindex=False))

if __name__ == "__main__":
    main()
