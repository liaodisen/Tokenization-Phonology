
from datasets import load_dataset, Dataset
import pandas as pd
import copy
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
from typing import Dict, Sequence
IGNORE_INDEX = -100


def preprocess(examples, tokenizer, model_name, max_length=512):
    if model_name == "llama3.1":
        instruction = f'{examples["instruction"]}\n{examples["words"]}'
        answer = examples["reasoning"]
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": answer + " The answer is " + examples["answer"]}
        ]
        formatted_sample = tokenizer.apply_chat_template(messages, 
                                                         tokenize=False)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_inputs = tokenizer(formatted_sample, 
                                 max_length=max_length, 
                                 truncation=True,
                                 padding="max_length", 
                                 return_tensors="pt",
                                 add_special_tokens=True)
        model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])
        return model_inputs
    elif model_name == "mT5":
        input_text = examples["instruction"]
        output_text = examples["answer"]
        inputs = tokenizer(input_text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        labels = tokenizer(output_text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        model_inputs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels.input_ids
        }
        return model_inputs
    

def preprocess_conversation(examples, tokenizer, model_name):
    instruction = examples["instruction"]
    output = examples["output"]
    word_selected = examples["selected words"]
    ipa_selected = examples["selected words ipa"]
    word_indices = examples["indices"]

    if model_name == "llama3.1":
        IPA_tokens = ['<IPA>', '</IPA>']
        IPA_special_tokens = {"additional_special_tokens": IPA_tokens}

        tokenizer.add_special_tokens(IPA_special_tokens)

        # insert IPA tokens around the words
        # find the index of the words in the instruction
        word_indices = [instruction.find(word) for word in word_selected]
        for word in word_selected:
            instruction = instruction.replace(word, f"{IPA_tokens[0]}{word}{IPA_tokens[1]}")

        if len(word_indices) == 0:
            ipa_sentence = None
        elif len(word_indices) == 1:
            ipa_sentence = f"{word_selected[0]} has IPA {ipa_selected[0]}\n"
        elif len(word_indices) == 2:
            ipa_sentence = f"{word_selected[0]} has IPA {ipa_selected[0]} and {word_selected[1]} has IPA {ipa_selected[1]}\n"
        instruction = f"{instruction}"
        if ipa_sentence is not None:
            answer = f"{ipa_sentence}{output}"
        else:
            answer = output
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": answer}
        ]
        formatted_sample = tokenizer.apply_chat_template(messages, tokenize=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_inputs = tokenizer(formatted_sample, 
                                 max_length=tokenizer.model_max_length, 
                                 truncation=True,
                                 return_tensors="pt",
                                 add_special_tokens=True)
        model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])
        return model_inputs
    else:
        # raise ValueError(f"Model {model_name} not supported")
        return None
    

def preprocess_phonetic(examples, tokenizer, model_name):
    if model_name == "llama3.1":
        instruction = examples["instruction"]
        output = examples["output"]
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        formatted_sample = tokenizer.apply_chat_template(messages, tokenize=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_inputs = tokenizer(formatted_sample, 
                                 max_length=tokenizer.model_max_length, 
                                 truncation=True,
                                 return_tensors="pt",
                                 add_special_tokens=True)
        model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])
        return model_inputs
    else:
        return None


def load_custom_dataset(path, tokenizer, model_name, max_length=512):
    dataset = load_dataset("json", data_files={"train": path})
    if model_name == "llama3.1":
        dataset = dataset["train"].map(
            preprocess,
            fn_kwargs={"tokenizer": tokenizer, "model_name": model_name, "max_length": max_length},
            remove_columns=['id', 'instruction', 'words', 'reasoning', 'answer']
        )
    elif model_name == "mT5":
        dataset = dataset["train"].map(
            preprocess,
            fn_kwargs={"tokenizer": tokenizer, "model_name": model_name, "max_length": max_length},
            remove_columns=['id', 'instruction', 'answer']
        )
    return dataset


def load_phonetic_dataset(path, tokenizer, model_name):
    dataset = load_dataset("json", data_files={"train": path})
    dataset = dataset["train"].map(
        preprocess_phonetic,
        fn_kwargs={"tokenizer": tokenizer, "model_name": model_name},
        remove_columns=[
            'instruction', 
            'output', 
        ]
    )
    return dataset


def load_conversation_dataset(path, tokenizer, model_name):
    dataset = load_dataset("json", data_files={"train": path})
    dataset = dataset["train"].map(
        preprocess_conversation,
        fn_kwargs={"tokenizer": tokenizer, "model_name": model_name},
        remove_columns=[
            'instruction', 
            'output', 
            'selected words', 
            'selected words ipa', 
            'indices',
        ]
    )
    return dataset


class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x).squeeze(0) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x).squeeze(0) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"additional_special_tokens": ["<IPA>", "</IPA>"]})
    # dataset = load_dataset("json", data_files={"train": "dataset/OpenHermes-2.5-filtered-2000.jsonl"})
    dataset = load_conversation_dataset("dataset/conversation_processed.jsonl", tokenizer, "llama3.1")
    print(tokenizer.decode(dataset[3]["input_ids"][0], skip_special_tokens=False))


