from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from trl import SFTTrainer
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PromptTuningConfig
)
from accelerate.utils import DistributedType
from utils import IPA_TOKENS
from dataset import (
    load_conversation_dataset, 
    load_phonetic_dataset, 
    DataCollatorForSupervisedDataset,
)
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from accelerate import Accelerator
# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")

@dataclass
class DataArguments:
    data_path_conversation: str = field(
        default=None, metadata={"help": "Path to the training dataset"}
    )
    data_path_phonetic: str = field(
        default=None, metadata={"help": "Path to the phonetic dataset"}
    )

@dataclass
class PeftArguments:
    use_lora: bool = False
    use_prompt_tuning: bool = False
    prompt_tuning_tokens_num: int = 10
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "Target modules for LoRA"}
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PeftArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        peft_args,
    ) = parser.parse_args_into_dataclasses()

    # Initialize Accelerator
    accelerator = Accelerator()
    training_args.gradient_checkpointing = False

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        llm_int8_threshold=0.0,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=peft_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    special_tokens = {"additional_special_tokens": IPA_TOKENS}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Apply LoRA configuration if specified
    if peft_args.use_lora:
        lora_config = LoraConfig(
            r=peft_args.lora_r,  # Rank of the low-rank adaptation
            lora_alpha=peft_args.lora_alpha,  # Scaling factor
            lora_dropout=peft_args.lora_dropout,  # Dropout rate for LoRA
            target_modules=peft_args.target_modules,  # Target modules for LoRA
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora_config).to(training_args.device)
    else:
        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=peft_args.prompt_tuning_tokens_num,
            tokenizer_name_or_path=model_args.model_name_or_path
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, peft_config).to(training_args.device)
    model.print_trainable_parameters()

    
    encoded_dataset_phonetic = load_phonetic_dataset(data_args.data_path_phonetic, tokenizer, "llama3.1")
    encoded_dataset_conversation = load_conversation_dataset(data_args.data_path_conversation, tokenizer, "llama3.1")
    first_element_phonetic = encoded_dataset_phonetic[2]
    first_element_conversation = encoded_dataset_conversation[2]
    print(tokenizer.decode(first_element_phonetic['input_ids'][0], skip_special_tokens=False))
    print(tokenizer.decode(first_element_conversation['input_ids'][0], skip_special_tokens=False))

    train_dataset = concatenate_datasets([encoded_dataset_phonetic, encoded_dataset_conversation])


    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    model_name = f"{model_args.model_name_or_path}_IPA_model"

    # muti-GPU
    # model, train_dataset, data_collator = accelerator.prepare(model, train_dataset, data_collator)


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,        
        data_collator=data_collator,
    )

    trainer.train()
    accelerator.wait_for_everyone()
    trainer.save_model(f"model/{model_name}")


if __name__ == "__main__":
    train()
