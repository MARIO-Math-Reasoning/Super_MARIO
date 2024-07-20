"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import re
import random
import json
from typing import List, Dict, Any, Optional, Type, Tuple, Union

from math_evaluation import is_equiv

from mcts_math.prompts.prompt_react import PROMPT_REACT
from mcts_math.prompts.prompt_sft import react_sft_prompt
from mcts_math.tools.python_tool import PythonInterpreter

from mcts_math.constants import *


python_tool_string = f"{PythonInterpreter().name}: {PythonInterpreter().description}"
python_tool_name = PythonInterpreter().name
    

# standard react for Round 1
def react_prompt_wrap(
    question: str, 
    partial_solution: str,
    config,
) -> str:
    step_delim = config.step_delim
    prompt_react = PROMPT_REACT(config)
    if partial_solution:
        inputs = f"{question}{step_delim}{partial_solution}{step_delim}"  
    else:
        inputs = f"{question}{step_delim}"  

    react_examples = prompt_react.random_examples()
    assert len(react_examples) > 0, "at least one example should be provided."

    if len(react_examples) > 1:
        example_prefix = "The following are %d demonstration examples." % len(react_examples)
    elif len(react_examples) == 1:
        example_prefix = "The following is a demonstration example."

    format_instructions = prompt_react.react_format_instructions.format(tool_desc=python_tool_string, tool_names=python_tool_name)

    prompt = "\n\n".join([format_instructions, example_prefix, *react_examples, prompt_react.react_suffix.format(input=inputs)])
    return prompt


def react_obs_wrap(observation: str) -> str:
    return f"{OBSERVATION}{observation}"


def react_step_result_unwrap(
    text: str,
    final_answer_action: str = FINAL_ANSWER_ACTION,
    action: str = ACTION,
    action_input: str = ACTION_INPUT,
) -> Tuple[str, Dict[str, str]]:
    includes_answer = final_answer_action in text
    regex = (
        r"{act}[\s]*(.*?)[\s]*{act_inp}[\s]*(.*)".format(act=action.strip(), act_inp=action_input.strip())
    )
    action_match = re.search(regex, text, re.DOTALL)

    parser_result = {
        "action": "",
        "action_input": "",
        "final_answer": "",
    }
    if action_match:
        if includes_answer:
            raise ValueError(
                f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
            )
        action = action_match.group(1).strip()
        action_input = action_match.group(2)
        tool_input = action_input.strip(" ").strip('"')

        parser_result["action"] = action
        parser_result["action_input"] = tool_input
        return text, parser_result

    elif includes_answer:
        parser_result["final_answer"] = text.split(final_answer_action)[-1].strip()
        return text, parser_result
    
    else:
        raise ValueError(f"Could not parse LLM output: `{text}`")


# SFT react for Round >1
def react_sft_prompt_wrap(
    question: str, 
    partial_solution: str,
    config, 
) -> str:
    step_delim = config.step_delim
    if partial_solution:
        inputs = f"{partial_solution}{step_delim}"
    else:
        inputs = f""

    prompt = react_sft_prompt.format(question=question, partial_solution=inputs)
    return prompt


def react_sft_obs_wrap(observation: str) -> str:
    return f"{OBSERVATION_LTAG}\n{observation}\n{OBSERVATION_RTAG}\n{STEP_RTAG}"


def react_sft_step_result_unwrap(
    text: str,
    final_answer_action: str = FINAL_ANSWER_ACTION,
) -> Tuple[str, Dict[str, str]]:
    includes_answer = final_answer_action in text
    regex = r"{code_ltag}[\s]*(.*[\s]```)".format(code_ltag=CODE_LTAG)  # CODE_LTAG  '<code>'
    action_match = re.search(regex, text, re.DOTALL)

    parser_result = { 
        "action": "",
        "action_input": "",
        "final_answer": "",
    }

    if action_match:
        if includes_answer:
            raise ValueError(
                f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
            )
        
        text = f"{text.strip()}\n{CODE_RTAG}"

        action = python_tool_name
        action_input = action_match.group(1)
        tool_input = action_input.strip(" ").strip('"')

        parser_result["action"] = action
        parser_result["action_input"] = tool_input
        return text, parser_result

    elif includes_answer:
        answer_regex = r'{faa}[\s]*(.*)</p>'.format(faa=final_answer_action)
        answer_match = re.search(answer_regex, text, re.DOTALL)
        if answer_match:
            parser_result["final_answer"] = answer_match.group(1).strip()
        else:
            parser_result["final_answer"] = text.split(final_answer_action)[-1].strip()
        
        parser_result["final_answer"] = remove_single_dollar(parser_result["final_answer"])
        
        text = f"{text.strip()}\n</step>"
        return text, parser_result
    
    else:
        raise ValueError(f"Could not parse LLM output: `{text}`")


def remove_single_dollar(s):
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s

def math_is_equiv(grt: Union[str, list[str]], prd: str):
    prd = remove_single_dollar(prd)
    if isinstance(grt, list):
        for g in grt:
            if is_equiv(remove_single_dollar(g), prd):
                return True
        return False
    else:
        return is_equiv(remove_single_dollar(grt), prd)
