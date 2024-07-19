import json
import random

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class PROMPT_REACT:
    def __init__(self, config):
        self.react_format_instructions = None
        self.react_suffix = None
        self.few_examples = None
        self.num_few_shot = config.num_few_shot
        self.load_prompt(config)

        assert self.num_few_shot <= len(self.few_examples), f"{config.num_few_shot} should be less than few_examples."   

    def load_prompt(self, config):

        self.few_examples = load_json(config.few_shot_path)
        prompt = load_json(config.prompt_path)
        self.react_format_instructions = prompt['react_format_instructions']
        self.react_suffix = prompt['react_suffix']

    def random_examples(self):
        selected_examples = random.sample(self.few_examples, min(len(self.few_examples), self.num_few_shot))
        return selected_examples
