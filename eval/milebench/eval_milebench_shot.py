import itertools
from argparse import ArgumentParser
import time
import json
import os
import random
import numpy as np
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from tqdm import tqdm

from typing import List, Dict

from eval.milebench.utils import MileBenchDataset as BaseMileBenchDataset
from internvl2_5.train.dataset import build_transform, dynamic_preprocess
from eval.mm_niah.tools import init_dist
from eval.mm_niah.eval_mm_niah import build_model

def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='dataset/benchmark/MileBench')
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--bsz', default=1, type=int)
    parser.add_argument('--combine_image', default=None, type=int, help='Use combined N images for evaluation.')
    # parser.add_argument('--model_configs', default='configs/model_configs.yaml')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num-gpus-per-rank', type=int, default=2)
    parser.add_argument('--max_context_len', type=int, default=512000)
    parser.add_argument('--n_tokens_per_image', type=int, default=256)
    parser.add_argument('--dynamic-image-size', action='store_true')
    parser.add_argument('--max-dynamic-patch', type=int, default=12)
    parser.add_argument('--resize-image', action='store_true')
    parser.add_argument('--rope_pos_id_version', type=str, default='default')
    parser.add_argument('--rope_pos_id_stride', type=int, default=None)
    parser.add_argument('--local_rank', type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument('--n-shot', type=int, default=0, 
                      help='Number of few-shot examples (0 for zero-shot)')
    parser.add_argument('--example-seed', type=int, default=42,
                      help='Random seed for example selection')
    parser.add_argument('--example-max-patch', type=int, default=4,
                      help='Max patches for example images')

    args = parser.parse_args()
    print(f'{args.checkpoint=}')
    print(f'{args.rope_pos_id_version=}')

    args.output_dir = os.path.join(args.output_dir, f"{args.dataset_name}")
    args.output_pth = os.path.join(args.output_dir, f"pred.json")

    return args

SIZE_MAP = {
    (320, 480): (420, 480),
    (266, 480): (420, 480),
    (480, 318): (480, 420),
    (480, 392): (480, 420),
    (360, 480): (420, 480),
    (480, 360): (480, 420),
    (392, 480): (420, 480),
    (480, 276): (480, 272),
    (480, 320): (480, 420),
    (480, 352): (480, 420),
    (480, 268): (480, 420),
    (1920, 1080): (1152, 648),
    (1280, 720): (1152, 648),
    (1920, 896): (1280, 600)
}

# ------------------ DynamicExamplePool 保持不变 ------------------
class DynamicExamplePool:
    def __init__(self, full_data: List[Dict], seed: int = 42, rank: int = 0):
        # 使用全局seed + 样本ID哈希保证跨rank一致性
        self.rng = np.random.default_rng(seed)
        self.rank = rank
        
        # 生成ID到样本的映射
        self.id2sample = {x['sample_id']: x for x in full_data}
        self.id2idx = {x['sample_id']: i for i, x in enumerate(full_data)}
        self.all_ids = list(self.id2idx.keys())
        
        # 预先生成所有样本的候选列表（排除自身）
        self.candidate_map = {
            sid: [xid for xid in self.all_ids if xid != sid]
            for sid in self.all_ids
        }

    def get_examples(self, current_id: int, n_shot: int) -> List[Dict]:
        candidates = self.candidate_map[current_id]
        
        # 根据全局seed和样本ID生成确定性的子种子
        sub_seed = self.rng.integers(0, 2**32) + self.id2idx[current_id]
        sub_rng = np.random.default_rng(sub_seed)
        
        selected_ids = sub_rng.choice(
            candidates, 
            size=min(n_shot, len(candidates)), 
            replace=False
        ).tolist()
        
        return [self.id2sample[xid] for xid in selected_ids]

# ------------------ Few-shot 数据集版本 ------------------
class MileBenchDataset(BaseMileBenchDataset):
    def __init__(self, example_pool=None, n_shot=0, **kwargs):
        # 注意：这里不再调用父类 __getitem__ 逻辑，而是自己构造，
        # 以保证 few-shot 的文本 prompt 和图片列表顺序一致。
        super().__init__(**kwargs)
        self.example_pool = example_pool
        self.n_shot = n_shot

    def __getitem__(self, index):
        # if self.n_shot == 0:
        # return super().__getitem__(index)
        ann = self.annotation[index]
        # 对当前样本调用统一处理（不做截断，保证图片顺序与文本一致）
        current_processed = self._process_annotation(ann)

        # 获取 few-shot 示例，并对每个示例进行同样处理
        examples = []
        if self.n_shot > 0:
            examples = self.example_pool.get_examples(ann['sample_id'], self.n_shot)
        processed_examples = [self._process_annotation(ex) for ex in examples]

        # 构造 few-shot prompt：
        # 每个示例块会显示 “Example Question i:”、对应数量的 <image> 占位符、示例文本和答案，
        # 最后再附上当前问题的文本（经过 _process_annotation 得到）。
        new_context = self._build_multi_shot_context(processed_examples, current_processed)

        # 合并图片路径：
        # 顺序为：先依次放入每个 few-shot 示例的图片（按 _process_annotation 中返回的顺序），再接上当前样本的图片。
        all_images = []
        is_example_flags = []
        for ex_processed in processed_examples:
            all_images.extend(ex_processed['raw_img_list'])
            is_example_flags.extend([True] * len(ex_processed['raw_img_list']))
        all_images.extend(current_processed['raw_img_list'])
        is_example_flags.extend([False] * len(current_processed['raw_img_list']))

        ret = {
            "sample_id": ann['sample_id'],
            "context": new_context,
            "raw_img_list": all_images,
            "response": str(ann['response'])
        }
        # 如果存在多选题选项，可附加到返回值中
        if 'choice_list' in ann['task_instance']:
            ret["choice_list"] = ann['task_instance']['choice_list']
        ret["is_example"] = is_example_flags
        return ret

    def collate_fn(self, batch):
        batch_data = super().collate_fn(batch)
        batch_data['is_example'] = [item['is_example'] for item in batch]
        return batch_data

    def _process_annotation(self, ann: dict) -> dict:
        """
        模拟父类 __getitem__ 中对 annotation 的处理逻辑，
        包括 task_instruction、choice_list、图片占位符替换以及构造 raw_img_list。
        这里不做截断处理，主要用于 few-shot 示例及当前样本 prompt 构造，
        同时保证返回的图片列表顺序与文本中的 <image> 占位符数量一致。
        """
        # 取出任务说明
        task_instruction = self.task_instructions[ann['task_instruction_id']]
        context = ann['task_instance']['context']

        # 处理多选题（choice_list）
        if 'choice_list' in ann['task_instance']:
            choice_str = '\nChoice list: \n'
            choice_str += '\n'.join([
                (f'{chr(65+idx)}. ' if self.dataset_name != 'GPR1200' else '') + f'{item}'
                for idx, item in enumerate(ann['task_instance']['choice_list'])
            ])
            choice_str += "\nAnswer with the option's letter from the given choices directly."
            context += choice_str

        # 替换图片占位符（保持与原逻辑一致）
        img_num = len(ann['task_instance']['images_path'])
        if self.combine_image:
            for i in range(img_num):
                rmv_txt = '{image#%d}' % (i+1)
                rmv_tbl = '{table#%d}' % (i+1)
                context = context.replace(rmv_txt, '<image> ')
                context = context.replace(rmv_tbl, '<image> ')
        else:
            for i in range(img_num):
                rmv_txt = '{image#%d}' % (i+1)
                rmv_tbl = '{table#%d}' % (i+1)
                context = context.replace(rmv_txt, '<image>')
                context = context.replace(rmv_tbl, '<image>')

        # 拼接任务说明：直接在最前面加上 instruction
        if self.combine_image:
            context = '<image>\n' + task_instruction + '\n' + context
        else:
            context = task_instruction + '\n' + context

        # 构造图片路径列表，与原版逻辑一致
        raw_img_list = []
        if self.combine_image:
            combine_image_str = f'combined_{self.combine_image}_images'
            for p in ann['task_instance'][combine_image_str]:
                img_path = os.path.join(self.img_dir.replace(os.path.basename(self.img_dir), combine_image_str), p)
                raw_img_list.append(img_path)
        else:
            for p in ann['task_instance']['images_path']:
                img_path = os.path.join(self.img_dir, p)
                if 's3:/' in p:
                    img_path = 'langchao2:' + p
                raw_img_list.append(img_path)

        return {"context": context, "raw_img_list": raw_img_list, "response": str(ann['response'])}

    def _build_multi_shot_context(self, processed_examples: List[dict], current_processed: dict) -> str:
        builder = []
        for idx, ex in enumerate(processed_examples, 1):
            builder.append(
                f"Example Question {idx}:\n{ex['context']}\nAnswer: {ex['response']}\n"
            )
        builder.append(f"Current Question:\n{current_processed['context']}")
        return "\n".join(builder)


# ------------------ 修改后的图像处理函数 ------------------
def load_image(image_file, dynamic_image_size=True, input_size=448, max_num=12, 
               return_additional_info=False, is_example=False, example_max_patch=12, args=None):
    
    def _process_image(img):
        """保持原有图像预处理逻辑"""
        if args.resize_image and img.size in SIZE_MAP:
            return img.resize(SIZE_MAP[img.size])
        return img

    if not return_additional_info:
        image = Image.open(image_file).convert('RGB')
        image = _process_image(image)
        
        # 对示例图像限制最大patch数量
        effective_max = min(max_num, example_max_patch) if is_example else max_num

        transform = build_transform(is_train=False, input_size=input_size)
        if dynamic_image_size:
            images = dynamic_preprocess(image, image_size=input_size, 
                                        use_thumbnail=True, 
                                        max_num=effective_max)
        else:
            images = [image]
        pixel_values = [transform(img) for img in images]
        return torch.stack(pixel_values)
    
    else:
        image = Image.open(image_file).convert('RGB')
        orig_size = image.size
        image = _process_image(image)
        
        effective_max = min(max_num, example_max_patch) if is_example else max_num

        transform = build_transform(is_train=False, input_size=input_size)
        if dynamic_image_size:
            images, boxes = dynamic_preprocess(image, image_size=input_size,
                                               use_thumbnail=True,
                                               max_num=effective_max,
                                               return_box=True)
        else:
            images = [image]
            boxes = [(0, 0, orig_size[0], orig_size[1]), ]
        
        pixel_values = [transform(img) for img in images]
        return torch.stack(pixel_values), images, boxes, orig_size

# ------------------ InferenceSampler 与 split_data、save 等辅助函数 ------------------
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

def split_data(data):
    '''
    按图片数量对数据进行分组
    ex: {
        2: [sample1, ...],
        3: [sample2, ...]
    }
    '''
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict

def save(results, accelerator, args):
    if accelerator.is_main_process:
        if os.path.exists(args.output_pth):
            if not args.overwrite:
                print(f'{args.output_pth} exists. Please pass `overwrite=True` to avoid unwanted overwriting.')
                exit(0)
        json.dump(results, open(args.output_pth, 'w'), ensure_ascii=False, indent=4)

# ------------------ 主函数 ------------------
def main(args):
    init_dist(args)

    task = args.dataset_name
    model_name = os.path.basename(args.checkpoint)
    model, tokenizer = build_model(args)

    print(f"Rank [{args.rank}] "
          f"Begin to eval model {args.checkpoint} on task {args.dataset_name}, "
          f"devices: {set([p.device for p in model.parameters()])}")
    
    if args.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    ######################### Loading Data #########################
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    combine_image = args.combine_image
    dataset_dir = os.path.join(data_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, 'images')

    # 加载完整数据集（用于构造 few-shot 示例池）
    core_file = (f'{dataset_name}_combined_{combine_image}.json' 
                if combine_image and combine_image != 1 else f'{dataset_name}.json')
    full_annotation = json.load(open(os.path.join(dataset_dir, core_file)))
    example_pool = DynamicExamplePool(
        full_data=full_annotation['data'],
        seed=args.example_seed,
        rank=args.rank
    )

    # 按图片数量分组数据，并排序（保持原有逻辑）
    data_dict = split_data(full_annotation['data'])
    data_dict = dict(sorted(data_dict.items(), key=lambda x: x[0], reverse=False))
    ################################################################

    ###################### Start Generating ########################
    print('Initialization Finished')
    print(f'Predicting {dataset_name} Using {model_name}')
    generation_config = dict(
        do_sample=False,
        num_beams=1,
        max_new_tokens=32,
    )
    outputs = []
    
    for n_img, sub_data in data_dict.items():
        print(f'Proceeding {n_img}-length images samples | Num: {len(sub_data)}')
        
        # 初始化数据集时传入 few-shot 示例池及 n_shot 参数
        lc_dataset = MileBenchDataset(
            annotation=sub_data,
            task_instructions=full_annotation['meta_data']['task_instruction'],
            img_dir=img_dir,
            max_context_len=args.max_context_len,
            n_tokens_per_image=args.n_tokens_per_image,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            combine_image=combine_image,
            example_pool=example_pool,
            n_shot=args.n_shot
        )

        lc_dataloader = DataLoader(
            dataset=lc_dataset,
            sampler=InferenceSampler(len(lc_dataset)),
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            collate_fn=lc_dataset.collate_fn
        )

        if args.rope_pos_id_stride is not None:
            rope_pos_id_stride = args.rope_pos_id_stride
        else:
            model_config = model.config
            rope_pos_id_stride = getattr(model_config, 'rope_pos_id_stride', None)
        print(f'USE {rope_pos_id_stride=}')

        for batch in tqdm(lc_dataloader) if args.rank == 0 else lc_dataloader:
            for _, (sample_id, question, images_list, gt_response) in enumerate(zip(
                batch['id'], batch['question'], batch['image_path'], batch['gt_response']
            )):
                all_boxes, num_tiles, for_posid_image_list, orig_sizes = [], [], [], []
                
                if len(images_list) > 0:
                    pixel_values = []
                    num_patches_list = []
                    # 根据 collate_fn, batch['is_example'] 为 [batch_size, list]，这里取 batch['is_example'][0]
                    for img, is_example in zip(images_list, batch['is_example'][0]):
                        curr_pixel_values, images, boxes, orig_size = load_image(
                            img,
                            dynamic_image_size=args.dynamic_image_size,
                            max_num=args.max_dynamic_patch,
                            return_additional_info=True,
                            is_example=is_example,
                            example_max_patch=args.example_max_patch,
                            args=args
                        )
                        for_posid_image_list.append(images)
                        all_boxes.append(boxes)
                        num_tiles.append(len(images))
                        orig_sizes.append(orig_size)
                        pixel_values.append(curr_pixel_values)
                        num_patches_list.append(len(curr_pixel_values))
                    
                    pixel_values = torch.cat(pixel_values)
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                else:
                    pixel_values = None
                    num_patches_list = []
                
                try:
                    if args.rope_pos_id_version == 'default':
                        pred = model.chat(
                            tokenizer=tokenizer,
                            pixel_values=pixel_values,
                            question=question,
                            num_patches_list=num_patches_list,
                            generation_config=generation_config
                        )
                    else:
                        pred = model.chat(
                            tokenizer=tokenizer,
                            pixel_values=pixel_values,
                            num_patches_list=num_patches_list,
                            question=question,
                            generation_config=generation_config,
                            num_tiles=[num_tiles, ],
                            all_boxes=[all_boxes, ],
                            orig_sizes=[orig_sizes, ],
                            image_list=[for_posid_image_list, ],
                            rope_pos_id_version=args.rope_pos_id_version,
                            rope_pos_id_stride=rope_pos_id_stride
                        )
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[Rank {args.rank}] OutOfMemoryError occurs! error: {e}")
                    pred = 'None'
                    torch.cuda.empty_cache()
                
                pred = pred.strip()
                print(f"[{current_time()}] [Rank {args.rank}], {pred=}, answer={gt_response}")
                outputs.append({
                    'sample_id': sample_id,
                    'question': question,
                    'gt_response': gt_response,
                    'gen_kwargs': dict(generation_config),
                    'pred_response': pred.strip(),
                })
    
    # 结果收集与保存
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    parsed_outputs = []
    for rank_data_str in merged_outputs:
        if rank_data_str is not None:
            parsed_outputs.extend(json.loads(rank_data_str))
    merged_outputs = parsed_outputs

    if args.rank == 0:
        json.dump(merged_outputs, open(args.output_pth, 'w'), ensure_ascii=False, indent=4)
        
        print(f'evaluating {task} ...')
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = os.path.join(args.output_dir, f'{task}_{time_prefix}.json')
        json.dump(merged_outputs, open(results_file, 'w'))
        print(f'Results saved to {results_file}')

        cmd_string = f'python eval/milebench/evaluate.py \
            --data-dir {args.data_dir} \
            --dataset {args.dataset_name} \
            --result-dir {args.output_dir}'
        print(cmd_string)
        os.system(cmd_string)
    
    torch.distributed.barrier()

if __name__ == '__main__':
    args = parse_args()
    main(args)
