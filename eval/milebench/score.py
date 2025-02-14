import json
import argparse
import os
import pandas as pd
FOLDER='milebench'
def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = round(sum(d[key] for d in dict_list) / len(dict_list), 5) * 100
    return mean_dict

def main(args):
    ########################## Set Dataset Taxonomy ##########################
    dataset_list = {
        'Realistic Temporal': [
            'ActionLocalization', 'ActionPrediction', 'ActionSequence', 'CharacterOrder',
            'CounterfactualInference', 'EgocentricNavigation', 'MovingAttribute', 'MovingDirection',
            'ObjectExistence', 'ObjectInteraction', 'ObjectShuffle', 'SceneTransition', 'StateChange'
        ],
        'Realistic Semantic': [
            'ALFRED', 'CLEVR-Change', 'DocVQA', 'IEdit', 'MMCoQA', 'MultiModalQA',
            'nuscenes', 'OCR-VQA', 'SlideVQA', 'Spot-the-Diff', 'TQA', 'WebQA', 'WikiVQA'
        ],
        'Diagnostic': ['TextNeedleInAHaystack', 'ImageNeedleInAHaystack', 'GPR1200']
    }

    ########################## Collect Evaluation Result ##########################
    result_dir = args.result_dir

    for model_name in args.models:
        print(f'\nProcessing {model_name}...')

        # Check if the result file already exists
        json_path = os.path.join(result_dir, FOLDER, model_name, 'milebench_result.json')
        if os.path.exists(json_path):
            print(f'Results for {model_name} already exist. Skipping...')
            continue

        print(f'Collecting results for {model_name}...')
        model_result = {}
        for task_name, dataset_names in dataset_list.items():
            task_result = {}
            if not dataset_names:
                continue

            for dataset in dataset_names:
                # print(f'Processing dataset: {dataset}')
                try:
                    eval_path = os.path.join(result_dir, FOLDER, model_name, dataset, 'eval.json')
                    if not os.path.exists(eval_path):
                        print(f'\t{model_name}--{dataset}  No evaluation file found')
                        task_result[dataset] = {}
                        continue

                    dataset_result = json.load(open(eval_path))
                except Exception as e:
                    print(eval_path)
                    print(f'Exception: {e}')
                    continue

                task_result[dataset] = dataset_result

            model_result[task_name] = task_result

        ########################## Save Result ##########################
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        json.dump(
            model_result,
            open(json_path, 'w'),
            ensure_ascii=False,
            indent=4
        )
        print(f'Results written to {json_path}')

        # Convert JSON to DataFrame & Save to CSV
        def parse_json_to_df(data):
            parsed_data = []
            try:
                for model, tasks in data.items():
                    model_data = {'Model': model}
                    for task, datasets in tasks.items():
                        for dataset, metrics in datasets.items():
                            # for metric, value in metrics.items():
                            #     if metric not in [
                            #         "image_quantity_level-Accuracy",
                            #         "image_quantity_level-Result",
                            #         "Diff-Accuracy"
                            #     ]:  # Ignore specific metrics
                            #         model_data[f"{dataset} ({metric})"] = round(value * 100, 2)
                            if isinstance(metrics, dict):  # 如果 metrics 是字典，处理键值对
                                for metric, value in metrics.items():
                                    if isinstance(value, (int, float)):  # 单一数值
                                        if metric not in [
                                            "image_quantity_level-Accuracy",
                                            "image_quantity_level-Result",
                                            "Diff-Accuracy"
                                        ]:  # 忽略特定指标
                                            model_data[f"{dataset} ({metric})"] = round(value * 100, 2)
                                    elif isinstance(value, dict):  # 处理嵌套字典
                                        for sub_metric, sub_value in value.items():
                                            if isinstance(sub_value, (int, float)):
                                                model_data[f"{dataset} ({metric}-{sub_metric})"] = round(sub_value * 100, 2)
                            elif isinstance(metrics, (int, float)):  # 如果 metrics 是浮点数或整数
                                model_data[f"{dataset}"] = round(metrics * 100, 2)
                            
                    parsed_data.append(model_data)
            except Exception as e:
                print(f"Exception encountered while processing data.")
                print(f"Current model: {model}")
                print(f"Current task: {task}")
                print(f"Current dataset: {dataset}")
                print(f"Current metrics: {metrics}")
                raise
            return pd.DataFrame(parsed_data)

        df = parse_json_to_df(model_result)
        csv_path = os.path.join(result_dir, FOLDER, model_name, 'milebench_result.csv')
        df.to_csv(csv_path, index=False)
        # print(f'CSV written to {csv_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, required=True)
    args = parser.parse_args()
    # args.models = [
    #     'internvl2_5_8b_stage1_mammoth-default',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride1',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride2',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride4',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride8',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride16',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride32',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride64',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride128',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride256']
    # args.models = [
    #     'internvl2_5_8b_stage1_mammoth-baseline-default',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride1',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride2',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride4',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride8',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride16',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride32',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride64',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride128',
    #     'internvl2_5_8b_stage1_mammoth-baseline-v2pe-stride256',]
    args.models = [
        'internvl2_5_8b_stage2_mammoth-baseline-v2pe-stride64',]
    # args.models = [
    #     'internvl2_5_8b',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride1',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride2',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride4',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride8',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride16',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride32',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride64',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride128',
    #     'internvl2_5_8b_stage1_mammoth_baseline-v2pe-stride256',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride1',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride2',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride4',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride8',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride16',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride32',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride64',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride128',
    #     'internvl2_5_8b_stage1_mammoth-v2pe-stride256'
    #     ]
    main(args)

'''
python eval/milebench/score.py \
    --result-dir /map-vepfs/yuansheng/LongContext/V2PE/eval_logs
'''