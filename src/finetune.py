from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig
from transformers.trainer_pt_utils import LabelSmoother
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from accelerate.utils import DistributedType
from utils import IPA_TOKENS
from dataset import (
    load_conversation_dataset, 
    load_phonetic_dataset, 
    DataCollatorForSupervisedDataset,
)
from torch.utils.data import DataLoader
from peft import PromptTuningConfig
from datasets import concatenate_datasets
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Meta-Llama/Meta-Llama-3.1-8B-Instruct")


@dataclass
class DataArguments:
    data_path_conversation: str = field(
        default=None, metadata={"help": "Path to the trainingdataset"}
    )
    data_path_phonetic: str = field(
        default=None, metadata={"help": "Path to the phonetic dataset"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        }
    )
    use_lora: bool = False
    use_wandb: bool = False

@dataclass
class PromptTuningArguments:
    prompt_tuning_tokens_num: int = 10

# @dataclass
# class LoraArguments:
#     lora_r: int = 64
#     lora_alpha: int = 16
#     lora_dropout: float = 0.05
#     lora_target_modules: List[str] = field(
#         default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
#     )
#     lora_weight_path: str = ""
#     lora_bias: str = "none"
#     q_lora: bool = False


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PromptTuningArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        prompt_tuning_args,
    ) = parser.parse_args_into_dataclasses()
    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )


    sepcial_tokens = {"additional_special_tokens": IPA_TOKENS}
    tokenizer.add_special_tokens(sepcial_tokens)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    if training_args.local_rank == 0:
        for st in sepcial_tokens["additional_special_tokens"]:
            print(f"{st}:{tokenizer.convert_tokens_to_ids(st)}")
    model.resize_token_embeddings(len(tokenizer))

    peft_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=prompt_tuning_args.prompt_tuning_tokens_num,
        tokenizer_name_or_path=model_args.model_name_or_path
    )
    model = get_peft_model(model, peft_config)

    if training_args.local_rank == 0:
        model.print_trainable_parameters()

    # encoded_dataset_conversation = load_conversation_dataset(data_args.data_path_conversation, tokenizer, "llama3.1")
    encoded_dataset_phonetic = load_phonetic_dataset(data_args.data_path_phonetic, tokenizer, "llama3.1")
    # combined_dataset = concatenate_datasets([encoded_dataset_conversation, encoded_dataset_phonetic])
    train_dataset = encoded_dataset_phonetic
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    if training_args.local_rank == 0:
        print(model.config)

    model_name = f"{model_args.model_name_or_path}_IPA_model"

    training_args.run_name = model_name
        
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f"model/{model_name}")



if __name__ == "__main__":
    train()