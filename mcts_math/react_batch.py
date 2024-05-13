"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os

from functools import partial
from typing import List, Any, Dict, Callable
from tqdm import tqdm
from pydantic import BaseModel, field_validator
from omegaconf import DictConfig, OmegaConf

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from pebble import ProcessPool
from concurrent.futures import TimeoutError

from .agents import REACT
from .constants import TIMEOUT_SECONDS


class REACTBatch(BaseModel):

    config: Any

    stop: List[str] = None
    
    llm: Callable[[...], List[RequestOutput]] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.config.stop:
            self.stop = OmegaConf.to_object(self.config.stop)

        self.llm = self.create_llm()

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            return cfg

        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")

    def create_llm(self):
        GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
        llm = LLM(
            model=self.config.model_dir, 
            tensor_parallel_size=len(GPUS), 
            trust_remote_code=True,
            seed=self.config.seed,
            swap_space=self.config.swap_space,
        )
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            use_beam_search=self.config.use_beam_search,
            best_of=self.config.best_of,
            max_tokens=self.config.max_tokens, 
            n=1,
            stop=self.stop,
            #seed=self.config.seed,
        )
        return partial(
            llm.generate,
            sampling_params=sampling_params,
        )

    @staticmethod
    def processor(solver: REACT, output: RequestOutput) -> REACT:
        step_result = output.outputs[0].text.strip()
        try:
            step_result, parser_result = solver.step_unwrap(step_result)
        except Exception as e:
            parser_result = None
        solver.update_current_node(step_result, parser_result)
        return solver
    
    def batch_generate(self, questions: List[str]):

        solvers = [REACT(config=self.config, question=question) for question in questions]

        for step in tqdm(range(self.config.max_depth), desc="Step Processing"):
            prompts = []
            epoch_solvers = []
            next_solvers = []

            for solver in solvers:
                if solver.should_generate_next():
                    prompts.append(solver.create_prompt())
                    epoch_solvers.append(solver)
                else:
                    next_solvers.append(solver)
            
            next_solver_span = len(next_solvers)
            if len(epoch_solvers) < 1:
                break
            
            # vllm run
            outputs = self.llm(prompts)
            # post-process outputs
            with ProcessPool(max_workers=min(len(epoch_solvers), os.cpu_count())) as pool:
                future = pool.map(self.__class__.processor, epoch_solvers, outputs, timeout=TIMEOUT_SECONDS)
                iterator = future.result()
            
            if len(epoch_solvers) > 100:  
                progress_bar = tqdm(total=len(epoch_solvers), desc="Execute")  
            else:  
                progress_bar = None 

            while True:
                try:
                    result = next(iterator)
                    next_solvers.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    next_solvers.append(None)
                    print(error)
                except Exception as error:
                    print(error)
                    next_solvers.append(None)
                if progress_bar is not None:
                    progress_bar.update(1) 
            
            if progress_bar is not None:
                progress_bar.close() 

            # update solvers
            assert len(epoch_solvers) == len(next_solvers[next_solver_span:]), f"Data is not matched, {len(epoch_solvers)} vs {len(next_solvers[next_solver_span:])}."
            for idx, (ori_solver, new_solver) in enumerate(zip(epoch_solvers, next_solvers[next_solver_span:])):
                if new_solver is None:
                    next_solvers[next_solver_span + idx] = ori_solver
            solvers = next_solvers

        jsonlines = {}
        for i, solver in enumerate(solvers):         
            jsonlines[solver.question] = solver.return_states()
        
        return jsonlines
