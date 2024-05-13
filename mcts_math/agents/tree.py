"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
from abc import abstractmethod
from termcolor import colored
from typing import Optional, Any, Dict, List, Callable, Type, Tuple, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from omegaconf import DictConfig, OmegaConf

from timeout_decorator import timeout

from mcts_math.config import BaseConfig
from mcts_math.nodes.base_node import BaseNode
from mcts_math.tools.python_tool import PythonInterpreter
from mcts_math.constants import TIMEOUT_SECONDS, TIMEOUT_MESSAGE, QUESTION_COLOR


def _python_ast_init():
    python = PythonInterpreter(globals=globals(), locals=None)
    return python


def tool_wrapper(tool):
    def _tool(query):
        return tool.run(query)
    return _tool


def no_action_wrapper(tool):
    def _tool(query):
        return "No action, no observation. Please continue to solve."
    return _tool


tools = {
    "python_interpreter": tool_wrapper(_python_ast_init()),
    "None": no_action_wrapper(_python_ast_init()),
}


class BaseTree(BaseModel):

    config: Any
    question: str

    ground_truth: Optional[Union[str, List[str]]] = None
    
    llm_model_id: str = None
    llm: Any = None

    root: Optional[Type[BaseNode]] = None
    current_node: Optional[Type[BaseNode]] = None 

    stop: Optional[List[str]] = None

    node_max_retry: int = 5

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.llm_model_id = self.config.model_dir

        if self.config.stop:
            # omegaconf.listconfig.ListConfig -> list
            self.stop = OmegaConf.to_object(self.config.stop)

        self.root = self.create_root()
        self.current_node = self.root

        if self.config.verbose and self.question:
            print(colored(f"Question: {self.question}\n", QUESTION_COLOR))

        if self.config.create_local_llm:
            self.llm = self.create_llm()
    
    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            if not os.path.exists(cfg.model_dir):
                raise ValueError(f"Model directory \"{cfg.model_dir}\" cannot be found.")
            return cfg

        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")
    
    def create_root(self) -> Type[BaseNode]:
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent: Optinal[Type[BaseNode]] = None) -> Type[BaseNode]:
        """
        subclass must implement
        """
    
    @abstractmethod
    def create_llm(self) -> Callable[[...], List[str]]:
        """
        subclass must implement
        """
    
    @abstractmethod
    def generate(self) -> None:
        """
        subclass must implement
        """
    
    def collect_partial_solution(self, node: Type[BaseNode]) -> str:
        # from leaf to root, and reverse
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            node = node.parent
        return self.config.step_delim.join(reversed(trajectory))
    
    def return_states(self) -> Dict[str, Dict[str, str]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            if node.has_children():
                candidates.extend(node.children)
        return states


def code_execution(
    node: Type[BaseNode], 
    parser_result: Dict[str, str],
) -> str:

    @timeout(TIMEOUT_SECONDS, exception_message=TIMEOUT_MESSAGE)
    def _code_execution(node: Type[BaseNode], parser_result: Dict[str, str]) -> str:
        # Define tool
        action = parser_result["action"]
        tool_func = tools[action]

        # Preventing variable update between different children
        # For each child, we re-run the historical code snippets with the same action (tool).
        history_action_inputs = collect_action_inputs(node, action)
        for history_ai in history_action_inputs:
            _ = tool_func(history_ai)
        
        # then, we execute current code snippets
        action_input = parser_result["action_input"]
        observation = str(tool_func(action_input)).strip()
        del tool_func
        return observation
    
    try:
        observation = _code_execution(node, parser_result)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))
    
    return observation


def collect_action_inputs(
    node: Type[BaseNode], 
    action: str,
) -> List[str]:
    action_inputs = []
    while node: 
        if node.state["action"] == action and \
            "TimeoutError" not in node.state["text"].split(node.state["action_input"])[-1]:
            action_inputs.append(node.state["action_input"])
        node = node.parent
    return action_inputs[::-1]