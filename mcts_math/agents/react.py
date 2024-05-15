"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import re

from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from functools import partial
from vllm import LLM, SamplingParams

from mcts_math.llms.local_llms import local_vllm
from mcts_math.nodes.base_node import BaseNode
from mcts_math.constants import (
    NO_VALID_CHILD,
    SOLUTION_COLOR,
    OBSERVATION_COLOR,
    WARNING_COLOR,
)

from .tree import BaseTree, code_execution


class REACT(BaseTree):

    REACT_NODE_KEYS: List[str] = ["action", "action_input", "final_answer"]
    prompt_wrap: Optional[Callable[[...], str]] = None
    obs_wrap: Optional[Callable[str, str]] = None
    step_unwrap: Optional[Callable[[...], Dict[str, str]]] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.config.prompt_wrap == "react":
            from .utils import react_prompt_wrap, react_obs_wrap, react_step_result_unwrap

            self.prompt_wrap = react_prompt_wrap
            self.obs_wrap = react_obs_wrap
            self.step_unwrap = react_step_result_unwrap

        elif self.config.prompt_wrap == "react_sft":
            from .utils import react_sft_prompt_wrap, react_sft_obs_wrap, react_sft_step_result_unwrap

            self.prompt_wrap = react_sft_prompt_wrap
            self.obs_wrap = react_sft_obs_wrap
            self.step_unwrap = react_sft_step_result_unwrap

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        super().validate_config(cfg)
        if not cfg.mode == "react":
            raise ValueError(f"Wrong value for config mode, must be react")
        if not cfg.n_generate_sample == 1:
            raise ValueError(f"Wrong value for config n_generate_sample, must be 1")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg
    
    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        return BaseNode(
            parent=parent, 
            additional_state_keys=self.REACT_NODE_KEYS,
        )

    def create_llm(self) -> Callable[[...], List[str]]:
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        llm = LLM(
            model=self.config.model_dir, 
            tensor_parallel_size=len(GPUS), 
            trust_remote_code=True,
            seed=self.config.seed,
            swap_space=self.config.swap_space,
        )
        sampling_params = SamplingParams(
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            use_beam_search=self.config.use_beam_search,
            best_of=self.config.best_of,
            max_tokens=self.config.max_tokens, 
            stop=self.stop,
            #seed=self.config.seed,
        )
        return partial(
            local_vllm,
            llm=llm,
            sampling_params=sampling_params,
            n=1,
            temperature=self.config.temperature,
        )

    def should_generate_next(self) -> bool:
        return not self.current_node.is_terminal and self.current_node.depth <= self.config.max_depth

    def generate(self) -> None:
        """
        generate as a linked list
        root -> x -> y -> z
        """
        while self.should_generate_next():
            step_result, parser_result = self.get_parsable_samples()
            self.update_current_node(step_result, parser_result)

    def update_current_node(
        self, 
        step_result: str,
        parser_result: Dict[str, str],
    ) -> None:
        self._update_current_node(step_result, parser_result)
        self.current_node = self.current_node.children[0]

    def _update_current_node(
        self,
        step_result: str,
        parser_result: Dict[str, str],
        idx: int = 0,
    ) -> None:
        if self.config.verbose:
            print(colored(f"{step_result}\n", SOLUTION_COLOR))

        # initialize a new node
        new_node = self.create_node(parent=self.current_node)
        new_node.tag = f"{self.current_node.tag}.{idx}"
        new_node.depth = self.current_node.depth + 1

        # update node state
        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
        elif parser_result["action"]:
            observation = code_execution(self.current_node, parser_result)
            observation = self.obs_wrap(observation)

            if self.config.verbose:
                    print(colored(f"{observation}\n", OBSERVATION_COLOR))

            new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
        else:
            if self.config.verbose:
                print(colored(f"WARNING: '{step_result}' Cannot resolve\n", WARNING_COLOR))
            new_node.state["text"] = step_result

        # update parent's children
        self.current_node.children.append(new_node)
    
    def get_parsable_samples(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = self.create_prompt()
        sampled_step_results = self.get_llm_samples(prompt)

        try:
            step_result = sampled_step_results[0]
            return self.step_unwrap(step_result)
        except Exception as e:
            n_samples = 3
            temperature = 0.7
            print(f"Exception: {e}. will retry {self.node_max_retry} times by setting temperature {temperature}, and generating {n_samples} samples in single run to save token counts")
        
            retry_cnt = 0
            while retry_cnt < self.node_max_retry:
                sampled_step_results = self.get_llm_samples(prompt, n_samples, temperature)
                for step_result in sampled_step_results:
                    try:
                        return self.step_unwrap(step_result)
                    except Exception as e:
                        retry_cnt += 1
                        print(f"Exception: {e}. Retry {retry_cnt} failed.")
                        continue
            return step_result, None

    def create_prompt(
        self,
    ) -> str:
        partial_solution = self.collect_partial_solution(self.current_node)
        prompt = self.prompt_wrap(
            self.question, 
            partial_solution,
            self.config,
        )
        return prompt
    
    def get_llm_samples(
        self, 
        prompt: str, 
        n: int = 1,
        temperature: Optional[float] = None,
    ) -> List[str]:
        if temperature is None:
            # default llm
            samples = self.llm(prompt, n=n)
        else:
            samples = self.llm(prompt, temperature=temperature, n=n)
        
        processed_samples = [sample.strip() for sample in set(samples)]
        return processed_samples
