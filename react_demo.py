import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse

from typing import List, Any, Dict
from omegaconf import OmegaConf

from mcts_math.agents import REACT
from mcts_math.config import BaseConfig


def load_qaf(filename: str) -> List[Dict[str, Any]]:
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
        if "example" in data:
            data = data["example"]
    elif filename.endswith(".jsonl"):
        data = []
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    else:
        raise ValueError(f"Unrecognized file format: {filename}")
    return data


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/react_demo.yaml")
    # input args
    args.add_argument('--question', type=str, default=None, help="your question")
    args.add_argument('--answer', type=str, default=None, help="ground truth")
    
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

    assert config.create_local_llm, f"Please set create_local_llm=True in config file."
    llm_version = os.path.basename(config.model_dir.rstrip("/"))

    if args.question:
        agent = REACT(config=config, question=args.question)
        
        agent.generate()
        # states = agent.return_states()
                
    else:
        raise ValueError("Need question or question file.")
