import os, sys
import os.path as osp
import torch
import torch.nn as nn
from transformers import AutoConfig
from modeling_value_head import ValueHead

model_name = "path to the pretrain model"
value_model_save_path = "path to the pretrain model-value_model"
os.makedirs(value_model_save_path, exist_ok=True)
os.system(f"cp -r {model_name} {value_model_save_path}")

config = AutoConfig.from_pretrained(model_name)
config.value_model = True

v_head = ValueHead(config)

# initialize
v_head.summary.weight.data.normal_(mean=0.0, std=config.initializer_range)
v_head.summary.bias.data.zero_()


# save file

config.save_pretrained(value_model_save_path)
v_head_state_dict = v_head.state_dict()
v_head_state_dict_with_prefix = {f'v_head.{k}': v for k, v in v_head_state_dict.items()}

torch.save(v_head_state_dict_with_prefix, osp.join(value_model_save_path, "value_head.pth"))