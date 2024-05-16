"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from typing import Any, Dict, Type, Optional, List
from pydantic import BaseModel
from omegaconf import OmegaConf
from tqdm import tqdm

from mcts_math.constants import (
    NO_VALID_CHILD, 
    TOO_MANY_STEPS, 
    TOO_MANY_CODE_ERRORS, 
)
from mcts_math.config import BaseConfig
from mcts_math.agents.utils import math_is_equiv


class InferNode(BaseModel):

    tag: str = "0"

    text: str = ""
    extra_info: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str = ""

    c_puct: float = 1.25
    depth: int = 0

    prior: float = 1.0
    value: float = 0
    q_value: float = 0
    visit_count: int = 0

    parent: Optional[Any] = None
    children: List[Any] = []

    prune: bool = False

    def puct(self) -> float:
        q_value = self.q_value if self.visit_count > 0 else 0
        u_value = self.c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + u_value


def rebuild_tree(
    tree_dict: Dict[str, Any], 
    max_num_children: int,
    c_puct: float,
    root_tag: str = "0",
) -> Tuple[Type[InferNoder], int]:
    root = InferNode(
        parent=None,
        tag=root_tag,
        c_puct=c_puct,
        **tree_dict[root_tag],
    )
    candidates = [root]
    max_depth = 0
    while candidates:
        node = candidates.pop(0)
        for idx in range(max_num_children):
            tag = f"{node.tag}.{idx}"
            depth = node.depth + 1
            if tag in tree_dict:
                child = InferNode(
                    parent=node,
                    tag=tag,
                    depth=depth,
                    c_puct=c_puct,
                    **tree_dict[tag],
                )
                max_depth = max(max_depth, depth)
                node.children.append(child)
                candidates.append(child)
    return root, max_depth


def is_valid_final_answer_node(node: Type[InferNode]) -> bool:
    if not node.children and node.final_answer and \
        node.final_answer not in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
        return True
    return False


def prune_node(node: Type[InferNode]) -> bool:
    if node.children:
        children_prune = []
        for child in node.children:
            children_prune.append(prune_node(child))
        if all(children_prune):
            node.prune = True
    else:
        # for leaf node
        if not is_valid_final_answer_node(node): 
            node.prune = True
    return node.prune


def select_non_prune(current_nodes: List[Type[InferNode]]) -> List[Type[InferNode]]:
        candidate_nodes = []
        for node in current_nodes:
            candidate_nodes.extend([child for child in node.children if not child.prune])
        return candidate_nodes


def sort_by_strategy(
    candidate_nodes: List[Type[InferNode]],
    strategy: str = "q_value",
) -> List[Type[InferNode]]:
    if strategy == "value":
        return sorted(candidate_nodes, key=lambda x: x.value, reverse=True)
    elif strategy == "q_value":
        return sorted(candidate_nodes, key=lambda x: x.q_value, reverse=True)
    elif strategy == "visit_count":
        return sorted(candidate_nodes, key=lambda x: x.visit_count, reverse=True)
    elif strategy == "puct":
        return sorted(candidate_nodes, key=lambda x: x.puct(), reverse=True)
    else:
        raise NotImplementedError(f"strategy {strategy} not implemented")

def get_solution( 
    full_tree_dict: Dict[str, Any], 
    prune: bool = False,
    b1: int = 1,
    b2: int = 5,
    strategy: str = "q_value",
    c_puct: float = 1.25,
) -> Optional[Dict[str, Any]]:
    """
    This function is used to extract solution from a built tree.
    It is mainly used for MCTS, but also works for saved tree from step_beam.
    """
    question = full_tree_dict["question"]
    ground_truth = full_tree_dict.get("answer", None)
    tree_dict = full_tree_dict["react"]

    # rebuild tree
    root, tree_depth = rebuild_tree(tree_dict, max_num_children=b1*b2, c_puct=c_puct)

    # pruning tree
    if prune:
        prune_node(root)
        if root.prune:
            # no valid leaf node for the entire tree
            return None
    
    # search in tree
    final_answer_nodes = []
    current_top_num = b1
    current_nodes = [root] 

    for _ in range(tree_depth):
        candidate_nodes = select_non_prune(current_nodes)
        candidate_nodes = sort_by_strategy(candidate_nodes, strategy)
        current_nodes = candidate_nodes[:current_top_num]

        for current_node in current_nodes[:]:
            if is_valid_final_answer_node(current_node):
                final_answer_nodes.append(current_node)
                current_nodes.remove(current_node)
                current_top_num -= 1
            elif not current_node.children:
                current_nodes.remove(current_node)
                current_top_num -= 1
    
    if not final_answer_nodes:
        return None

    final_answer_nodes = sort_by_strategy(final_answer_nodes, strategy)
    top_final_answer_node = final_answer_nodes[0]

    # for node in final_answer_nodes:
    #     print(node.tag)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": top_final_answer_node.final_answer,
        "tag": top_final_answer_node.tag,
    }


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--custom_cfg", type=str, default="configs/offline_inference.yaml")
    args.add_argument("--tree_jsonl", type=str, required=True, default="saved tree jsonl file.")

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


    cnt, total = 0, 0
    with open(args.tree_jsonl, "r") as f:
        for line in tqdm(f):
            full_tree_dict = json.loads(line)
            solution = get_solution(
                full_tree_dict,
                prune=config.prune,
                b1=config.step_beam_width,
                b2=config.n_generate_sample,
                strategy=config.mcts_infer_strategy,            
            )

            if solution and math_is_equiv(solution["ground_truth"], solution["final_answer"]):
                cnt += 1
            total += 1

    print(cnt, total, f"Accuracy: {cnt / total}")
