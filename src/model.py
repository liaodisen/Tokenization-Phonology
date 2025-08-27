from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration
from peft import get_peft_model
from datasets import load_dataset
from utils import RHYMING_PROMPTS
from torch.utils.data import DataLoader
import copy
import random
from peft import PromptTuningConfig, PromptTuningInit
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import torch
import numpy as np
import wandb
from dataset import load_custom_dataset, load_conversation_dataset
from sklearn.model_selection import train_test_split
import bitsandbytes as bnb
from exp.Prompt import prompt_g2p_simple
from utils import IPA_TOKENS    
from dataset import DataCollatorForSupervisedDataset

def preprocess_batch(batch, model_name, tokenizer, max_length=512):

    if model_name == "llama3.1":
        messages = [
            {"role": "user", "content": batch["prompt"]},
            {"role": "assistant", "content": batch["answer"]}
        ]
        formatted_sample = tokenizer.apply_chat_template(messages, 
                                                         tokenize=False)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
        prompt = batch["prompt"]
        answer = batch["answer"]

        model_inputs = tokenizer(prompt+answer, 
                                 max_length=max_length, 
                                 truncation=True,
                                 padding="max_length", 
                                 return_tensors="pt")
        model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])
        return model_inputs

def preprocess_func(example):
    prompt = random.choice(RHYMING_PROMPTS)
    example["prompt"] = prompt.format(word1=example["word1"], word2=example["word2"])
    answer = example["label"]
    if answer == True:
        answer = "This is a phonological question, need to check the IPA of the words. The word {word1} has IPA {ipa1} and the word {word2} has IPA {ipa2}, they have the same ending phoneme, therefore they are in rhyme. The answer is Yes."
    else:
        answer = "This is a phonological question, need to check the IPA of the words. The word {word1} has IPA {ipa1} and the word {word2} has IPA {ipa2}, they have different ending phonemes, therefore they are not in rhyme. The answer is No."
    example["answer"] = answer.format(word1=example["word1"], word2=example["word2"], ipa1=example["word1 IPA"], ipa2=example["word2 IPA"])
    return example


def generate_response(prompt, model, tokenizer, device, model_name="llama3.1"):
    if model_name == "llama3.1":
        messages = [
            {"role": "user", "content": prompt}
        ]
        formatted_sample = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(formatted_sample, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif model_name == "mT5":
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

class PromptTuning:
    def __init__(self, model_name, dataset_path, use_wandb=False, max_length=512, batch_size=4):
        if model_name == "llama3.1":
            self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.model_name = "llama3.1"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        elif model_name == "mT5":
            self.model_id = "google/mt5-large"
            self.model_name = "mT5"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        sepcial_tokens = {"additional_special_tokens": IPA_TOKENS}
        self.tokenizer.add_special_tokens(sepcial_tokens)
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        )
        for st in sepcial_tokens["additional_special_tokens"]:
            print(f"{st}:{self.tokenizer.convert_tokens_to_ids(st)}")
        self.model.resize_token_embeddings(len(self.tokenizer))
        encoded_dataset = load_conversation_dataset(dataset_path, self.tokenizer, model_name)
        encoded_dataset = encoded_dataset.train_test_split(test_size=10, seed=42)
        self.train_dataset = encoded_dataset["train"]
        self.val_dataset = encoded_dataset["test"]
        data_collator = DataCollatorForSupervisedDataset(self.tokenizer)
        self.data_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=data_collator
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=1, 
            shuffle=False,
            collate_fn=data_collator
        )
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="prompt-tuning")

    def set_optimizer(self, 
                     lr, 
                     num_train_epochs, 
                     weight_decay, 
                     warmup_steps):
        
        if hasattr(self.model, 'peft_config'):
            print("Model is initialized as a PEFT model.")
        else:
            print("Model is not a PEFT model.")

        self.optimizer = AdamW(self.model.parameters(), 
                               lr=lr, 
                               weight_decay=weight_decay)
        self.num_train_epochs = num_train_epochs
        self.scheduler = get_scheduler(
            name="linear", 
            optimizer=self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_train_epochs * len(self.data_loader)
        )


    def set_model(self):
        if self.model_name == "llama3.1":       
            peft_config = PromptTuningConfig(
                task_type="CAUSAL_LM",
                num_virtual_tokens=10,
                tokenizer_name_or_path=self.model_id
            )
            # self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.half()
            self.model = get_peft_model(self.model, peft_config)
            # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        elif self.model_name == "mT5":
            # peft_config = PromptTuningConfig(
            #     task_type="SEQ_2_SEQ_LM",
            #     num_virtual_tokens=50,
            #     tokenizer_name_or_path="google/mt5-base"
            # )
            self.model = MT5ForConditionalGeneration.from_pretrained(
                "google/mt5-large"
            )
            # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
            # self.model = get_peft_model(self.model, peft_config)

    def validate(self, device):
        loss_list = []
        for batch in self.val_dataloader:
            batch = {k: v.squeeze(dim=1).to(device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            if isinstance(loss, torch.Tensor):
                loss = loss.mean()  
            loss_list.append(loss.item())

        print(f"Validation loss: {np.mean(loss_list)}")
        return np.mean(loss_list)

    def train(self, device):
        self.model.to(device)
        self.model.train()

        for epoch in range(self.num_train_epochs):
            loss_list = []
            for step, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                batch = {k: v.squeeze(dim=1).to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                if isinstance(loss, torch.Tensor):
                    loss = loss.mean()
                loss_list.append(loss.item())
                loss.backward()  # Scale the loss and call backward
                self.optimizer.step()    # Step the optimizer
                self.scheduler.step()
                self.optimizer.zero_grad()
                if self.use_wandb:
                    wandb.log({"loss": np.mean(loss_list), 
                               "epoch": epoch, 
                               "learning rate": self.scheduler.get_last_lr()[0]})
                print(f"Epoch {epoch+1} loss: {np.mean(loss_list)}")

            val_loss = self.validate(device)
            if self.use_wandb:
                wandb.log({"validation loss": val_loss, "epoch": epoch})
            # prompt = prompt_g2p_simple.format(word="fur")
            # response = generate_response(prompt, self.model, self.tokenizer, device, self.model_name)
            # print(response)

    def test_dataloader(self, num_batches=1):
        for i, batch in enumerate(self.data_loader):
            if i >= num_batches:
                break
            print(f"Batch {i+1}:")
            x = batch["input_ids"].squeeze(dim=1)
            # Decode the first sequence in the batch
            decoded_text = self.tokenizer.decode(x[0].tolist(), skip_special_tokens=False)
            print("Decoded Text:", decoded_text)
            print("\n")

    def save_model(self, save_path):
        # If model is wrapped in DataParallel, get the underlying model
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model_to_save.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.1")
    parser.add_argument("--dataset", type=str, default="dataset/OpenHermes-2.5-filtered-2000.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()
    cls = PromptTuning(args.model, args.dataset, batch_size=args.batch_size, use_wandb=args.use_wandb)
    cls.set_model()
    cls.set_optimizer(
        lr=args.lr, 
        num_train_epochs=args.num_train_epochs, 
        weight_decay=args.weight_decay, 
        warmup_steps=args.warmup_steps)
    cls.train(device="cuda")
    cls.save_model(f"model/{args.model}_IPA_model")


