import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import numpy as np
import torch
from data_utils import CAT_SHORT2LONG, process_single_sample
from datasets import concatenate_datasets, load_dataset
from internvl2_5.model.internvl_chat import InternVLChatModel
from internvl2_5.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

cache_dir = "/map-vepfs/yuansheng/tmp/MMMU"

ds_collections = {
    'MMMU_validation': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'validation'
    },
    'MMMU_test': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'test'
    },
    'MMMU_dev': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'dev'
    },
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    options = [_['option'] for _ in batches]
    # 如果返回 few shot 标记，可额外返回 is_example 字段
    is_examples = [_['is_example'] for _ in batches] if 'is_example' in batches[0] else None
    return pixel_values, questions, answers, data_ids, options, is_examples

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

###############################################
# 新增：Few shot 示例池
###############################################
class DynamicExamplePool:
    def __init__(self, full_data, seed=42):
        self.rng = np.random.default_rng(seed)
        self.id2sample = {d['id']: d for d in full_data}
        self.id2idx = {d['id']: idx for idx, d in enumerate(full_data)}
        self.all_ids = list(self.id2sample.keys())
        self.candidate_map = {
            sid: [xid for xid in self.all_ids if xid != sid]
            for sid in self.all_ids
        }

    def get_examples(self, current_id, n_shot):
        candidates = self.candidate_map[current_id]
        current_idx = self.id2idx[current_id]
        sub_seed = int(self.rng.integers(0, 2**32)) + current_idx
        sub_rng = np.random.default_rng(sub_seed)
        selected_ids = sub_rng.choice(
            candidates,
            size=min(n_shot, len(candidates)),
            replace=False
        ).tolist()
        return [self.id2sample[sid] for sid in selected_ids]


###############################################
# 修改后的 MMMUDataset：增加 few shot 逻辑和 example_max_patch 设置
###############################################
class MMMUDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, n_shot=0, example_pool=None, example_max_patch=4):
        # run for each subject
        sub_dataset_list = []
        for subject in tqdm(CAT_SHORT2LONG.values(), desc="Loading subjects"):
            sub_dataset = load_dataset(root, subject, split=split, cache_dir=cache_dir)
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.data = concatenate_datasets(sub_dataset_list)
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        # few shot 参数
        self.n_shot = n_shot
        self.example_pool = example_pool
        self.example_max_patch = example_max_patch

    def __len__(self):
        return len(self.data)

    def _process_sample(self, data, is_example=False):
        """
        将 process_single_sample 处理后的数据进一步转换为模型所需格式，
        包括构造问题文本、图片处理及选项文本等。
        is_example 为 True 时，使用 example_max_patch 限制首个图片的最大 patch 数量。
        """
        data_id = data['id']
        question = data['question'].strip()
        pil_images = data['image']
        question_type = data['question_type']

        choices = eval(data['options'])
        answer = data['answer'] if 'answer' in data else None

        choice_list = []
        options = {}
        multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c.strip()))
            options[multiple_choices[i]] = c.strip()
        choice_txt = '\n'.join(choice_list)

        if self.dynamic_image_size:
            images = []
            for idx, pil_image in enumerate(pil_images):
                if pil_image is not None:
                    if idx == 0:
                        pil_image = pil_image.resize((pil_image.width * 2, pil_image.height * 2), Image.BILINEAR)
                        # 如果是 few shot 示例，限制最大 patch 数量
                        effective_max = min(self.max_num, self.example_max_patch) if is_example else self.max_num
                        pil_imgs = dynamic_preprocess(pil_image, image_size=self.input_size,
                                                      use_thumbnail=self.use_thumbnail, max_num=effective_max)
                    else:
                        pil_imgs = dynamic_preprocess(pil_image, image_size=self.input_size,
                                                      use_thumbnail=self.use_thumbnail, max_num=1)
                    images += pil_imgs
        else:
            images = [pil_images[0]]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # 将选项文本追加到问题后面
        if len(choice_txt) > 0:
            question += '\n' + choice_txt
        question += '\n' + self.prompt[question_type]
        question = question.strip()

        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
            'option': options,
            'data_id': data_id,
            'num_images': pixel_values.shape[0]  # 图片数目
        }

    def __getitem__(self, idx):
        # 处理当前样本，当前样本不做 example 限制
        current_data = process_single_sample(self.data[idx])
        current_sample = self._process_sample(current_data, is_example=False)

        # 若设置 few shot，则从示例池中获取 few shot 示例并构造 multi-shot prompt
        if self.n_shot > 0 and self.example_pool is not None:
            few_shot_raw = self.example_pool.get_examples(current_sample['data_id'], self.n_shot)
            few_shot_samples = []
            for raw in few_shot_raw:
                # 注意：为了避免数据泄漏，few shot 示例中可以排除答案（根据实际需求调整）
                processed = self._process_sample(process_single_sample(raw), is_example=True)
                few_shot_samples.append(processed)
            # 构造 few shot 文本（每个示例块格式为："Example Question i:" ...）
            prompt_lines = []
            for i, ex in enumerate(few_shot_samples, 1):
                prompt_lines.append(f"Example Question {i}:\n{ex['question']}\nAnswer: {ex['answer']}\n")
            prompt_lines.append(f"Current Question:\n{current_sample['question']}")
            combined_question = "\n".join(prompt_lines)

            # 拼接图片：few shot 示例图片在前，当前样本图片在后
            pixel_values_list = []
            is_example_flags = []
            for ex in few_shot_samples:
                pixel_values_list.append(ex['pixel_values'])
                is_example_flags.extend([True] * ex['num_images'])
            pixel_values_list.append(current_sample['pixel_values'])
            is_example_flags.extend([False] * current_sample['num_images'])
            combined_pixel_values = torch.cat(pixel_values_list, dim=0)

            return {
                'question': combined_question,
                'pixel_values': combined_pixel_values,
                'answer': current_sample['answer'],
                'option': current_sample['option'],
                'data_id': current_sample['data_id'],
                'is_example': is_example_flags
            }
        else:
            return current_sample


###############################################
# evaluate_chat_model 主函数（仅修改 default 分支）
###############################################
def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k
    return pred


def evaluate_chat_model():
    prompt = {
        'multiple-choice': "Answer with the option's letter from the given choices directly.",
        'open': 'Answer the question using a single word or phrase.'
    }
    random.seed(args.seed)

    for ds_name in args.datasets:
        # 先构建数据集（传入 n_shot 和 example_max_patch 参数），并利用全部数据构造 few shot 示例池
        dataset = MMMUDataset(
            root=ds_collections[ds_name]['root'],
            split=ds_collections[ds_name]['split'],
            prompt=prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            n_shot=args.n_shot,  # few shot 数量
            example_max_patch=args.example_max_patch
        )
        # 构造示例池：利用数据集中所有样本（转换为 list）来构建
        dataset.example_pool = DynamicExamplePool(dataset.data, seed=args.seed)

        if args.rope_pos_id_version == 'default':

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=InferenceSampler(len(dataset)),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=partial(collate_fn, tokenizer=tokenizer),
            )

            outputs = []
            for _, (pixel_values, questions, answers, data_ids, options, is_examples) in tqdm(enumerate(dataloader)):
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                generation_config = dict(
                    num_beams=args.num_beams,
                    max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                    min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                )
                # 此处 questions[0] 已经包含了 few shot 示例信息
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config
                )
                if len(options[0]) == 0:
                    preds = [pred]
                else:
                    preds = [post_process(pred, options[0])]

                for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
                    outputs.append({
                        'question': question,
                        'answer': pred,
                        'gt_answers': answer,
                        'data_id': data_id
                    })
        else:
            # 非 default 分支（rope_pos_id_version != 'default'）暂不做 few shot 修改，保持原逻辑
            def newposid_collate_fn(batches, tokenizer):
                pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
                questions = [_['question'] for _ in batches]
                answers = [_['answer'] for _ in batches]
                data_ids = [_['data_id'] for _ in batches]
                options = [_['option'] for _ in batches]

                num_tiles = [_['num_tiles'] for _ in batches]
                all_boxes = [_['all_boxes'] for _ in batches]
                orig_sizes = [_['orig_sizes'] for _ in batches]
                image_list = [_['image_list'] for _ in batches]
                num_patches_list = [_['num_patches_list'] for _ in batches]
                return pixel_values, questions, answers, data_ids, options, num_tiles, all_boxes, orig_sizes, image_list, num_patches_list

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=InferenceSampler(len(dataset)),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=partial(newposid_collate_fn, tokenizer=tokenizer),
            )

            if args.rope_pos_id_stride is not None:
                rope_pos_id_stride = args.rope_pos_id_stride
            else:
                model_config = model.config
                rope_pos_id_stride = getattr(model_config, 'rope_pos_id_stride', None)
            print(f'USE {rope_pos_id_stride=}')

            outputs = []
            for _, (pixel_values, questions, answers, data_ids, options, num_tiles, all_boxes, orig_sizes, image_list, num_patches_list) in tqdm(enumerate(dataloader)):
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                generation_config = dict(
                    num_beams=args.num_beams,
                    max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                    min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature)
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    num_tiles=num_tiles,
                    all_boxes=all_boxes,
                    orig_sizes=orig_sizes,
                    image_list=image_list,
                    rope_pos_id_version=args.rope_pos_id_version,
                    num_patches_list=num_patches_list[0],
                    rope_pos_id_stride=rope_pos_id_stride
                )
                if len(options[0]) == 0:
                    preds = [pred]
                else:
                    preds = [post_process(pred, options[0])]

                for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
                    outputs.append({
                        'question': question,     # 包含 few shot 拼接后的内容
                        'answer': pred,
                        'gt_answers': answer,
                        'data_id': data_id,
                        'image_list': image_list  # 将图片列表也保存下来
                    })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            output_path = os.path.join(args.out_dir, results_file)
            outputs_dict = {}
            for item in merged_outputs:
                outputs_dict[item['data_id']] = item['answer']
            with open(output_path, 'w') as f:
                json.dump(outputs_dict, f, indent=4)
            print('Results saved to {}'.format(output_path))
            if ds_collections[ds_name]['split'] == 'validation':
                print('Evaluating ...')
                cmd = f'python eval/mmmu/main_eval_only.py ' \
                      f'--output_path {output_path} ' \
                      f'--answer_path eval/mmmu/answer_dict_val.json'
                print(cmd)
                os.system(cmd)
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.jsonl'
            output_path = os.path.join(args.out_dir, results_file)
            with open(output_path, 'w') as writer:
                for item in merged_outputs:
                    writer.write(json.dumps(item) + '\n')
            print('Results saved to {}'.format(output_path))


###############################################
# 主函数及参数设置
###############################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='MMMU_validation')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--rope_pos_id_version', type=str, default='default')
    parser.add_argument('--rope_pos_id_stride', type=int, default=None)
    # 新增 few shot 参数
    parser.add_argument('--n_shot', type=int, default=0, help="Number of few-shot examples")
    parser.add_argument('--example-max-patch', type=int, default=4, help="Max patches for example images")
    args = parser.parse_args()
    print(f'{args.rope_pos_id_version=}')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, **kwargs).eval()
    if args.rope_pos_id_version !='default':
        # 此处非 default 分支的 __getitem__ 保持原逻辑
        if args.rope_pos_id_version != 'default':
            def __getitem__(self, idx):
                # 原始数据处理
                data = process_single_sample(self.data[idx])
                data_id = data['id']
                question = data['question'].strip()
                # pil_images 是一个 PIL 图片对象列表
                pil_images = data['image']
                question_type = data['question_type']

                choices = eval(data['options'])
                answer = data['answer'] if 'answer' in data else None

                choice_list = []
                options = {}
                multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
                for i, c in enumerate(choices):
                    choice_list.append('{}. {}'.format(multiple_choices[i], c.strip()))
                    options[multiple_choices[i]] = c.strip()
                choice_txt = '\n'.join(choice_list)

                if len(choice_txt) > 0:
                    question += '\n' + choice_txt
                question += '\n' + self.prompt[question_type]
                question = question.strip()

                # 图像处理（和原来一样），但这里 image_list 只保存图片“路径”
                num_tiles, all_boxes, orig_sizes, image_list, num_patches_list = [], [], [], [], []
                record_num_image = 0
                if self.dynamic_image_size:
                    images = []
                    for idx_img, pil_image in enumerate(pil_images):
                        if pil_image is not None:
                            if idx_img == 0:
                                pil_image = pil_image.resize((pil_image.width * 2, pil_image.height * 2), Image.BILINEAR)
                                orig_sizes.append(pil_image.size)
                                pil_image, boxes = dynamic_preprocess(pil_image, image_size=self.input_size,
                                                                    use_thumbnail=self.use_thumbnail, max_num=self.max_num, return_box=True)
                                num_tiles.append(len(pil_image))
                                all_boxes.append(boxes)
                                # 构造伪路径：例如 "dataid_img_0"
                                image_list.append([f"{data_id}_img_{i}" for i in range(len(pil_image))])
                                num_patches_list.append(len(pil_image))
                            else:
                                orig_sizes.append(pil_image.size)
                                pil_image, boxes = dynamic_preprocess(pil_image, image_size=self.input_size,
                                                                    use_thumbnail=self.use_thumbnail, max_num=1, return_box=True)
                                num_tiles.append(len(pil_image))
                                all_boxes.append(boxes)
                                image_list.append([f"{data_id}_img_{len(image_list)}"])
                                num_patches_list.append(len(pil_image))
                            images += pil_image
                            record_num_image += 1
                else:
                    images = [pil_images[0]]
                    orig_size = pil_images[0].size
                    orig_sizes.append(orig_size)
                    image_list.append([f"{data_id}_img_0"])
                    num_tiles.append(1)
                    all_boxes.append([(0, 0, orig_size[0], orig_size[1])])
                    num_patches_list.append(1)
                pixel_values = [self.transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)

                # 当设置了 few shot 时，增加 few shot 逻辑
                if self.n_shot > 0 and self.example_pool is not None:
                    few_shot_raw = self.example_pool.get_examples(data_id, self.n_shot)
                    few_shot_samples = []
                    few_shot_img_path_list = []  # 保存 few shot 示例的图片路径列表
                    for raw in few_shot_raw:
                        processed = self._process_sample(process_single_sample(raw), is_example=True)
                        few_shot_samples.append(processed)
                        num_imgs = processed.get('num_images', 1)
                        few_shot_img_path_list.append([f"{raw['id']}_img_{i}" for i in range(num_imgs)])
                    prompt_lines = []
                    for i, ex in enumerate(few_shot_samples, 1):
                        prompt_lines.append(f"Example Question {i}:\n{ex['question']}\nAnswer: {ex['answer']}\n")
                    prompt_lines.append(f"Current Question:\n{question}")
                    combined_question = "\n".join(prompt_lines)
                    current_img_paths = [p for sublist in image_list for p in sublist]
                    few_shot_paths = [p for sublist in few_shot_img_path_list for p in sublist]
                    combined_image_path_list = few_shot_paths + current_img_paths
                    question = combined_question
                else:
                    combined_image_path_list = [p for sublist in image_list for p in sublist]

                return {
                    'question': question,           # 包含 few shot 示例的完整问题文本
                    'pixel_values': pixel_values,
                    'answer': answer,
                    'option': options,
                    'data_id': data_id,
                    'num_tiles': num_tiles,
                    'all_boxes': all_boxes,
                    'orig_sizes': orig_sizes,
                    'image_list': combined_image_path_list,  # 这里只保存图片路径的列表
                    'num_patches_list': num_patches_list
                }
            MMMUDataset.__getitem__ = __getitem__

    if not args.load_in_8bit and not args.auto:
        model = model.cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')
    print(f'[test] n_shot: {args.n_shot}')
    print(f'[test] example_max_patch: {args.example_max_patch}')

    evaluate_chat_model()
