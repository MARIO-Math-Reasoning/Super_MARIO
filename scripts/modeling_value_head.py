# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput


KVCache = Tuple[torch.Tensor, torch.Tensor]


class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):

        if hidden_states.device != self.summary.weight.device:
            hidden_states = hidden_states.to(self.summary.weight.device)

        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        values = torch.tanh(output).squeeze(-1)
        return values


class AutoModelForCausalLMWithValueHead(nn.Module):
    r"""
    An autoregressive model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, `push_to_hub` and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the `ValueHead` class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                `ValueHead` class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                `ValueHead` if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                `ValueHead`. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the `ValueHead` with a random distribution. This is the default
                    strategy.
                - **"normal"** -- Initializes the weights of the `ValueHead` with a normal distribution.

    """
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.pretrained_model.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
    
    def _split_kwargs(self, kwargs):
        """
        Separate the kwargs from the arguments that we support inside
        `supported_args` and the ones that we don't.
        """

        supported_kwargs = {}
        unsupported_kwargs = {}
        peft_kwargs = {}

        for key, value in kwargs.items():
            if key in self.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

        return supported_kwargs, unsupported_kwargs, peft_kwargs

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        self.pretrained_model.load_weights(model_name_or_path, cache_dir,
                                            load_format, revision)

        # params_dict = dict(self.v_head.named_parameters())
        # for name, loaded_weight in hf_model_weights_iterator(
        #         model_name_or_path, cache_dir, load_format, revision):
        #     if name.strip("v_head.") in params_dict:
        #         param = params_dict[name.strip("v_head.")]
        #         weight_loader = getattr(param, "weight_loader",
        #                                 default_weight_loader)
        #         weight_loader(param, loaded_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        # cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.pretrained_model.model(input_ids, positions, kv_caches, input_metadata)  # cache_events
        # values = self.pretrained_model.v_head(hidden_states)

        # for idx, next_token in enumerate(next_tokens):
        #     next_token.prompt_logprobs = values[idx][-1].item()   

        return hidden_states


    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.pretrained_model.logits_processor(self.pretrained_model.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.pretrained_model.sampler(logits, sampling_metadata)
        return next_tokens

