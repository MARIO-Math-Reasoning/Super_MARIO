import os, sys
import os.path as osp
import time
import numpy as np
import pynvml

from multiprocessing import Process
from vllm import LLM, SamplingParams

import logging
logger = logging.getLogger(__name__)

TIMEOUT_PROCESS = 1800
TIME_SPAN_LLM = 0.5
MAX_SOLUTIONS_COUNT = 24

# LLM check time
CHECK_INTERVAL = 60 # 1800  # half hour
UNIT = 1024**3

BAR_TIME = 30 # 30


def llm_init(config):
    GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    llm = LLM(
        model=config.model_dir, 
        tensor_parallel_size=len(GPUS), 
        trust_remote_code=True,
        seed=config.seed,
        swap_space=config.swap_space
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        use_beam_search=config.use_beam_search,
        best_of=config.best_of,
        max_tokens=config.max_tokens, 
        n=config.n_generate_sample,
        stop=config.stop,
        #seed=config.seed,
    )
    return llm, sampling_params


def llm_engine(config):
    llm, sampling_params = llm_init(config)
    return llm, sampling_params


def _asyn_llm_engine(config, public_prompts, public_outputs, public_n, task_flag):
    try:
        llm, sampling_params = llm_init(config)

        while True:
            time.sleep(TIME_SPAN_LLM)

            task_key = []
            prompts = []
            n_list = []

            # generator
            for key, val in public_prompts.items():
                task_key.append(key)
                prompts.append(val)
                n_list.append(public_n[key])
                
            if len(prompts) > 0:
                task_flag.value = len(prompts)
                sampling_params.n = max(n_list) if len(n_list) > 0 else 1 
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                for key, output in zip(task_key, outputs):
                    # task finished
                    del public_prompts[key]
                    del public_n[key]

                    samples, prior_probs = [], []
                    for item in output.outputs:
                        samples.append(item.text)
                        prior_probs.append(0 if len(item.token_ids)==0 else np.exp(item.cumulative_logprob / len(item.token_ids)))

                    public_outputs[key] = {"texts": samples, "prior_probs": prior_probs, 'value': output.value_estimate}

    except Exception as e:
        logger.exception(f"llm error {e}", exc_info=True)


def get_all_gpu_memory_usage():
    pynvml.nvmlInit()  # init NVML
    # gpu_count = pynvml.nvmlDeviceGetCount() 
    gpu_memory_info = []
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    for i in available_gpus:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(i))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_info.append({
            'index': i,
            'total': mem_info.total,
            'used': mem_info.used,
            'free': mem_info.free
        })
    pynvml.nvmlShutdown()  # close NVML
    return gpu_memory_info


def asyn_llm_engine(config, public_prompts, public_outputs, public_n, task_flag, monitor_flag):
    """
    with Manager() as manager:
        public_prompts = manager.dict()  # input text for generator
        public_outputs = manager.dict()  # genrated text, avg prob and value from generator: {"texts": [], "prior_probs": [], "value": float}
        public_n = manager.dict()  # the number of generated text
        task_flag = manager.Value('i', 1)
        monitor_flag = manager.Value('i', 1)

        public_mcts_outputs = manager.dict()

        monitor_process = Process(target=monitor_llm_process, args=(args, public_prompts, public_outputs, public_n, task_flag, monitor_flag))
        monitor_process.start()

        ...

        logger.info("all question have been sampled.")
        monitor_flag.value = 0
        monitor_process.join()
    """
    llm_process = Process(target=_asyn_llm_engine, args=(config, public_prompts, public_outputs, public_n, task_flag))
    llm_process.start()
    time.sleep(CHECK_INTERVAL * 2)  # wait llm start
    while monitor_flag.value:
        time.sleep(CHECK_INTERVAL) 
        if monitor_flag.value == 0:
            print("break")
            break
        gpu_memory_info = get_all_gpu_memory_usage()

        for gpu in gpu_memory_info:
            if gpu['used'] / UNIT < 10:  # GPU memory less than 10GB
                logger.info('LLM process might be stuck. Restart required.')
                llm_process.terminate()
                llm_process.join()
                llm_process = Process(target=_asyn_llm_engine, args=(config, public_prompts, public_outputs, public_n, task_flag))
                llm_process.start()
            else:
                logger.info('LLM process is alive.')

    logger.info("finish llm_process")
    llm_process.terminate()
    llm_process.join()
