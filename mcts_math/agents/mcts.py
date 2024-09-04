"""
author: lmp-decaderan
email: ldecaderan@gmail.com

reviewed: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import random
import torch
import numpy as np

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from functools import partial
from pydantic import field_validator

from vllm.outputs import CompletionOutput, RequestOutput

from mcts_math.agents.utils import math_is_equiv as is_equiv

from mcts_math.nodes import MCTSNode
from mcts_math.constants import (
    TOO_MANY_CODE_ERRORS, 
    TOO_MANY_STEPS, 
    NO_VALID_CHILD, 
    SOLUTION_COLOR, 
    OBSERVATION_COLOR,
    WARNING_COLOR,
)

from .tree import BaseTree, code_execution
from .step_beam import SBSREACT


class MCTS(SBSREACT):

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "mcts":
            raise ValueError(f"Wrong value for config mode.")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg

    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent, 
            additional_state_keys=self.REACT_NODE_KEYS,
            c_puct=self.config.c_puct,
        )

    def generate(self) -> None:
        self.search()

    @torch.inference_mode()
    def search(self) -> None:
        for idx in range(self.config.iterations):
            # node selection starting from root
            node = self.selection()
            # expansion_evaluation_backpropagation
            self.expansion_evaluation_backpropagation(node)

    def selection(self) -> Optional[Type[MCTSNode]]:
        node = self.root
        while node.has_children() or node.is_terminal:
            next_node = self.select_child(node)     # To encourage exploration, select from non-terminal children
            if next_node is None:                   # if None，it mean all children are terminal
                node.is_terminal = True
                break
            node = next_node
    
        return None if node.is_terminal else node

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        # TODO: implement multi-strategy
        # select the best child according to the puct
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue

            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        return random.choice(best_childs) if best_childs else None

    def expansion_evaluation_backpropagation(self, node: Type[MCTSNode]) -> None:
        """
        This function is only used for single example inference, required to set `create_local_llm` as True.
        """
        assert self.config.create_local_llm, "llm must be created within MCTS class."
        prompt = self.create_prompt()
        # expand and evaluate
        outputs, value_estimate = self.llm(prompt, n=self.n_generate_sample, stop=self.stop)
        if value_estimate is not None:  # input exceeds 4096, output '' and None
            self.expand_node(outputs, node)
        else:
            value_estimate = self.config.negative_reward
            node.is_terminal = True
        # backup
        node.update_recursive(value_estimate, self.root)

    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
        if self.config.remove_duplicate:
            dedup_outputs = []
            dedup_keys = set()
            for output in outputs:
                key = output.text.strip()
                if not key in dedup_keys:
                    dedup_keys.add(key)
                    dedup_outputs.append(output)
            outputs = dedup_outputs
        for idx, output in enumerate(outputs):
            prior_prob = np.exp(output.cumulative_logprob / len(output.token_ids))
            step_result, parser_result = self.step_unwrap(output.text.strip())
            self.create_child(step_result, parser_result, node, prior_prob, idx)

    def create_child(
        self, 
        step_result: str, 
        parser_result: Dict[str, str], 
        node: Type[MCTSNode],
        prior_prob: float,
        idx: int,
    ) -> None:
        if self.config.verbose:
            print(colored(f"{step_result}\n", SOLUTION_COLOR))
        
        # initialize a new node
        new_node = self.create_node(parent=node)
        new_node.tag = f"{node.tag}.{idx}"
        new_node.depth = node.depth + 1
        new_node.prior = prior_prob

        # update node state
        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["action"]:
            observation = code_execution(node, parser_result)
            observation = self.obs_wrap(observation)

            if self.config.verbose:
                print(colored(f"{observation}\n", OBSERVATION_COLOR))

            new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]

            if "Error" in observation:
                new_node.consecutive_errors = node.consecutive_errors + 1
                if new_node.consecutive_errors > self.config.errors_threshold:
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                    self.eval_final_answer(new_node)
        else:
            if self.config.verbose:
                print(colored(f"WARNING: '{step_result}' Cannot resolve\n", WARNING_COLOR))
            new_node.state["text"] = step_result

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            node.update_recursive(self.config.negative_reward, self.root)
            return 
        
        if self.ground_truth:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer)
            # backup
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
        else:
            # for testset, no ground_truth, put this node in candidate_nodes, then it will be evaluated by value model and backup in select_next_step().
            self.candidate_nodes.append(node)

    def select_next_step(self, outputs: Optional[List[RequestOutput]] = None) -> None:
        """process output from vllm
        e.g.,
        prompts = tree.create_prompt(is_value_only=True)
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.current_nodes = []
        if outputs is not None:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                # assert self.question in output.prompt
                # backup
                if candidate_node.is_terminal and self.ground_truth:
                    continue
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True
                candidate_node.update_recursive(value_estimate, self.root)
                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        selection_node = self.selection()
        if selection_node is not None:
            self.current_nodes.append(selection_node)
    
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            # assert self.question in output.prompt
            # current_step.value = output.value
            # expand n_generate_sample nodes
            value_estimate = output.value_estimate
            if value_estimate is not None:  # input exceeds 4096, output '' and None
                current_node.value = value_estimate
                self.expand_node(output.outputs, current_node)
            else:
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True
            # self.expand_node(output.outputs, current_node)
            # self.candidate_nodes.extend(current_node.children)

            # backup
            if self.config.update_leaf_value:
                # child node will be put into candidate_nodes, then all candidate_nodes will be evaluated by value model and backup in select_next_step().
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node)
            else:
                current_node.update_recursive(value_estimate, self.root)

    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["prior"] = node.prior
            states[node.tag]["visit_count"] = node.visit_count()
            if node.has_children():
                candidates.extend(node.children)
        return states
