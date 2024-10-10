# The key implementation details based on [Llama_Factory v0.6.1](https://github.com/hiyouga/LLaMA-Factory/tree/v0.6.1).

## dataprocess
* We construct the corresponding `Q`-values for each steps
* `Q` is the same length as `input_ids`, and any position that is not the end of a step is marked with `IGNORE_INDEX` (typically -100).
```python
def preprocess_value_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X ` and labels with format `X <eos> `
    model_inputs = {"input_ids": [], "attention_mask": [], "Q": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:  # prompt only one, response only one
            continue

        message = examples["prompt"][i] + [{"role": 'assistant', 'content': ""}]
        input_ids, _ = template.encode_oneturn(tokenizer, message, examples["system"][i], examples["tools"][i])

        if data_args.train_on_prompt:
            print("train_on_prompt")
            source_mask = input_ids
            Q = [IGNORE_INDEX] * len(input_ids)

        else:
            source_mask = [IGNORE_INDEX] * len(input_ids)
            Q = [IGNORE_INDEX] * len(input_ids)

        # input_ids += source_ids + target_ids
        # labels += source_mask + target_ids
        labels = source_mask
        
        multistep_response = json.loads(examples["response"][i][0]['content'])
        response_state = multistep_response[-1]['Q']  # last Q
        for sub_response in multistep_response:
            if len(sub_response['step']) == 0:
                print(sub_response['step'])
            sub_message = [{"role": 'user', 'content': ""}] + [{"role": 'assistant', 'content': sub_response['step'].strip() + "\n"}]
            sub_Q = float(sub_response['Q'])
            _, sub_response_ids = template.encode_oneturn(tokenizer, sub_message, examples["system"][i], examples["tools"][i])
            
            # sub_response_ids = sub_response_ids[:-1]  # discard the 1000001
            # to make sure the sentence ends with \n instead of <eos>
            # our value model predicts the v based on '\n'

            input_ids += sub_response_ids
            Q += [IGNORE_INDEX] * (len(sub_response_ids) - 1) + [sub_Q]
            labels += sub_response_ids

            if len(input_ids) > data_args.cutoff_len:
                break

        if template.efficient_eos:  # vanilla template will go into
            input_ids += [tokenizer.eos_token_id]
            Q += [IGNORE_INDEX]
            labels += [tokenizer.eos_token_id]

        if len(input_ids) > data_args.cutoff_len:
            input_ids = input_ids[:data_args.cutoff_len]
            Q = Q[:data_args.cutoff_len]
            labels = labels[:data_args.cutoff_len]


        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["Q"].append(Q)


        if response_state == -1:
            model_inputs["labels"].append([IGNORE_INDEX] * len(labels))
        elif response_state == 1:
            model_inputs["labels"].append(labels)
        else:
            assert False, response_state

    return model_inputs
```


## DataCollator
* We inherit `DataCollatorForSeq2Seq` to organize `Q` into batches.
```python
@dataclass
class VMDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

            # Padding Q to the longest sequence in the batch
            if "Q" in features[0]:   # float32
                # max_length_Q = max(len(feature["Q"]) for feature in features)
                for feature in features:
                    remainder = [IGNORE_INDEX] * (max_label_length - len(feature["Q"]))  # Assuming IGNORE_INDEX as padding value for Q
                    
                    if isinstance(feature["Q"], list):
                        feature["Q"] = (
                            feature["Q"] + remainder if padding_side == "right" else remainder + feature["Q"]
                        )
                    elif padding_side == "right":
                        feature["Q"] = np.concatenate([feature["Q"], remainder]).astype(np.float32)
                    else:
                        feature["Q"] = np.concatenate([remainder, feature["Q"]]).astype(np.float32)

        features = pad_without_fast_tokenizer_warning(  # only padding input_ids and attention_mask
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
```

# models
* For training, we did not make any modifications to the package `trl`, so we need to pass the logits output by the model through the tanh activation function.
* To prevent loss `NaN`, we calculate the SFT loss outside the transformer.
```python
def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, torch.Tensor], return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        
        # Compute rewards
        Q = inputs.get("Q", None)
        if Q is not None:
            del inputs["Q"]
        
        labels = inputs.get("labels", None)
        if labels is not None:
            del inputs["labels"]
            
        mask = Q.ne(IGNORE_INDEX)

        lm_logits, loss, values = model(**inputs, output_hidden_states=True, return_dict=True)
        values = torch.tanh(values)

        if loss is None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if torch.all(shift_labels==IGNORE_INDEX):
                loss_fct = CrossEntropyLoss(reduction='sum')
            else:
                loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, model.pretrained_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        assert not torch.isnan(loss) and Q is not None

        Q = Q.type_as(values)
        masked_values = torch.where(mask, values, Q)
        value_loss = F.mse_loss(masked_values, Q, reduction='sum') / (mask.sum() + 1e-3)
        all_losses = loss + self.weight_alpha * value_loss


        if return_outputs:
            return all_losses, [all_losses, loss, value_loss, masked_values, Q]
        return all_losses, value_loss
```


## Model Loader
We will make sure to use `AutoModelForCausalLMWithValueHead` from `trl` for training through setting `add_valuehead=True` in `./src/llmtuner/model/loader.py` of Llama-Factory-v0.6.1.
```python
def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model. Must after load_tokenizer.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    if is_trainable and model_args.use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore

        unsloth_kwargs = {
            "model_name": model_args.model_name_or_path,
            "max_seq_length": model_args.model_max_length,
            "dtype": model_args.compute_dtype,
            "load_in_4bit": model_args.quantization_bit == 4,
            "token": model_args.hf_hub_token,
            "device_map": {"": get_current_device()},
            "rope_scaling": getattr(config, "rope_scaling", None),
        }
        try:
            model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
        except NotImplementedError:
            logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
            model_args.use_unsloth = False

        if model_args.adapter_name_or_path:
            model_args.adapter_name_or_path = None
            logger.warning("Unsloth does not support loading adapters.")

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **init_kwargs)

    patch_model(model, tokenizer, model_args, is_trainable)
    register_autoclass(config, model, tokenizer)

    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    if add_valuehead:  # We will set add_valuehead = True
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        config.value_model = True  # New add

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
        for param in model.parameters():
            if param.device.type == "cuda":
                param.data = param.data.to(model_args.compute_dtype)
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:d}".format(all_param)
    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model
```

## Other details
```python
tokenizer.padding_side = "left"
# Prevent discarding Q in the batch. 
training_args.remove_unused_columns = False 
```

* We limit the maximum length of training sequences to 1024.
* We did not try smaller epochs because we wanted to make the learning rate decrease more smoothly.

Please feel free to ask if you have any further questions.
