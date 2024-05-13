import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse

from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from mcts_math.react_batch import REACTBatch
from mcts_math.config import BaseConfig
from react_demo import load_qaf


def batch(iterable, n=-1):
    l = len(iterable)
    if n <= 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/react_sft.yaml")
    args.add_argument(
        "--qaf", "--question-answer-file", 
        type=str, 
        required=True,
        help="the file includes quesiton / partial solution (optional) / answer (optional)")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    print(config)

    assert config.model_dir, f"batch inference only for local ckpt"
    llm_version = os.path.basename(config.model_dir.rstrip("/"))

    data = load_qaf(args.qaf)   

    saved_jsonl_file = f"{args.qaf}.{llm_version}.jsonl" 
    with open(saved_jsonl_file, "w") as writer:
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            questions = [d["question"] for d in cur_data]
            agent = REACTBatch(config=config)
            jsonlines = agent.batch_generate(questions)
            for d in cur_data:
                question = d["question"]
                d["react"] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()