import os
import argparse
import json
from tqdm import tqdm

from mcts_math.agents.utils import math_is_equiv


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--res_file', type=str, required=True, help="result file in jsonl.")
    args.add_argument('--react', action="store_true", help="if using react_batch_demo.py, set True.")

    args = args.parse_args()
    return args


def eval_jsonl(res_file: str, react: bool = False) -> float:
    cnt, total = 0, 0
    with open(res_file, "r") as f:
        for line in tqdm(f):
            d = json.loads(line.strip())
            answer = d["answer"]
            if react:
                # react is the only one solution.
                last_tag = sorted(d["react"].keys(), key=lambda x: len(x), reverse=True)[0]
                prediction = d["react"][last_tag]["final_answer"]
            else:
                if d["react"]["solutions"]:
                    # get top-1
                    prediction = d["react"]["solutions"][0]["final_answer"]
                else:
                    prediction = ""
            if math_is_equiv(answer, prediction):
                cnt += 1
            total += 1
    return cnt / total


if __name__ == '__main__':
    args = parse_args()

    # you run the agent with qaf, including question and answer, e.g., `run_sbs.sh`
    # the output res_file will include question, answer and model prediction with top-1 value. 
    acc = eval_jsonl(args.res_file, args.react)
    print(acc)
