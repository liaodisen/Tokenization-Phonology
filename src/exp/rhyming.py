"""
This script is trying to extract the ARPAbet using the LLM
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
import transformers
from Prompt import *
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from Prompt import *
import re


def extract_answer(response):
    match = re.search(r"Answer:\s*(.+)", response)
    return match.group(1) if match else None


def apply_template(word1, word2, template, tokenizer, lora=False, input_type='word'):
    if input_type == 'word':
        word1 = word1
        word2 = word2
    elif input_type == 'slash':
        word1 = '/'.join(list(word1))
        word2 = '/'.join(list(word2))


    if lora:
        prompt = template.format(word1='<IPA>' + word1 + '</IPA>', word2='<IPA>' + word2 + '</IPA>')
    else:
        prompt = template.format(word1=word1, word2=word2)
    chat = [
        {"role": "user", "content": prompt}
    ]
    formatted_sample = tokenizer.apply_chat_template(chat, tokenize=False)
    return formatted_sample
    

def extract_answer_template(response, tokenizer):
    chat = [
        {"role": "user", "content": "Here is the model's response:"},
        {"role": "assistant", "content": response},
        {"role": "user", "content": f"From the above response, summarize the answer in the form of 'Yes' or 'No'."}
    ]
    formatted_sample = tokenizer.apply_chat_template(chat, tokenize=False)
    return formatted_sample

def create_model(model_name):
    if model_name == 'llama2':
        model_id = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == 'llama3.1':
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif model_name == 'llama3':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == 'mistral':
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    else:
        raise ValueError("Invalid model choice. Choose from 'llama2', 'llama3', or 'mistral'.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model


def generate_response(prompt, model, tokenizer, device, max_new_tokens=256):
    """Generate a response using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    return response.strip()


if __name__ == "__main__":
    access_token = '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'
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Generate rhyming words for nonsense words.')
    parser.add_argument('--input_file', type=str, default='dataset/rhyming_test.csv', help='Input file with nonsense words.')
    parser.add_argument('--LLM', type=str, choices=['llama3', 'mistral', 'yi', 'falcon', 'llama3.1', 'qwen', 'lora-llama3.1'], default='llama3.1', help='Choose between LLaMA 2, LLaMA 3, and Mistral.')
    parser.add_argument('--prompt_template', type=str, choices=['simple2', 'icl', 'simple', 'split'], default='simple', help='Choose between COT and ICL.')
    parser.add_argument('--input_type', type=str, choices=['word', 'slash'], default='word', help='Choose between word and slash.')
    parser.add_argument('--peft', action='store_true', help='Use PEFT.')
    args = parser.parse_args()

    if args.LLM == 'llama3':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.LLM == 'mistral':
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.LLM == 'llama3.1':
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.LLM == 'yi':
        model_id = "01-ai/Yi-1.5-6B-Chat"
    elif args.LLM == 'falcon':
        model_id = "tiiuae/Falcon3-7B-Instruct"
    elif args.LLM == 'qwen':
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    elif args.LLM == "lora-llama3.1":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        lora_model_path = "model/meta-llama/Meta-Llama-3.1-8B-Instruct_IPA_model"
        
        # Load the base model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        IPA_TOKENS = ['<IPA>', '</IPA>']
        special_tokens = {"additional_special_tokens": IPA_TOKENS}
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )
        
        # Load the base model with appropriate dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        
        # Load the LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_model_path)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # We'll use direct generation instead of vLLM for the LoRA model
        use_direct_generation = True
    else:
        raise ValueError("Invalid model choice. Choose from 'llama2', 'llama3', or 'mistral'.")
    
    # Only initialize tokenizer and vLLM if not using LoRA
    if args.LLM != "lora-llama3.1":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if args.LLM == 'mistral':
            llm = LLM(
                model=model_id,
                tokenizer_mode='mistral',
                load_format="mistral",
                config_format="mistral",
            )
        else:
            llm = LLM(model=model_id)

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=256,
            stop_token_ids=[tokenizer.eos_token_id],
            skip_special_tokens=True
        )

    if args.prompt_template == 'simple':
        prompt_template = prompt_template_simple2_rhyming
    
    LLM = args.LLM
    if args.LLM == 'lora-llama3.1':
        LLM = 'llama3.1'  # Use base model name for file paths
        
    input_file = f'dataset/rhyming_pairs2.csv'
    input_df = pd.read_csv(input_file)
    single_prompts = []
    
    for idx, row in input_df.iterrows():
        word1 = row['word1']
        word2 = row['word2']
        label = row['label']
        prompt_single = apply_template(word1, word2, prompt_template, tokenizer, lora=True, input_type=args.input_type)
        single_prompts.append(prompt_single)
    
    # Generate responses based on the model type
    if args.LLM == "lora-llama3.1":
        # Direct generation for LoRA model
        labels_single = []
        for prompt in tqdm(single_prompts, desc="Processing word pairs"):
            response = generate_response(prompt, model, tokenizer, device, max_new_tokens=256)
            single_answer = extract_answer(response)
            print(response, single_answer)
            labels_single.append(single_answer)
    else:
        # vLLM generation for other models
        single_responses = llm.generate(single_prompts, sampling_params=sampling_params)
        
        labels_single = []
        for single_response in single_responses:
            single_response = single_response.outputs[0].text
            single_response_cleaned = single_response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
            single_answer = extract_answer(single_response)
            labels_single.append(single_answer)
    
    input_df['prediction'] = labels_single
    output_file = f"results/rhyming_{args.prompt_template}_{args.LLM}_{args.input_type}.csv"
    input_df.to_csv(output_file, index=False)
    
    
    
    
    


