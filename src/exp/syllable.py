"""
This script is trying to extract the ARPAbet using the LLM
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
import transformers
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from Prompt import *
import re

def extract_syllables(response):
    """
    Extracts the syllables from the response.
    The expected format is: 'Header text... Answer: 2'
    """
    # match the number after Answer:
    match = re.search(r"Answer:\s*(\d+)", response)
    return match.group(1) if match else None



def apply_template(word, template, tokenizer, lora=False):
    if lora:
        prompt = template.format(word='<IPA>' + word + '</IPA>')
    else:
        prompt = template.format(word=word)
    chat = [
        {"role": "user", "content": prompt}
    ]
    formatted_sample = tokenizer.apply_chat_template(chat, tokenize=False)
    return formatted_sample
    

def extract_answer_template(response, tokenizer):
    chat = [
        {"role": "user", "content": "Here is the model's response:"},
        {"role": "assistant", "content": response},
        {"role": "user", "content": f"From the above response, give the ARPAbet transcription of the word, just the ARPAbet transcription, no other text."}
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
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
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
    print(response.strip())
    return response.strip()


if __name__ == "__main__":
    access_token = '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'
    use_llama2 = False  # Set to True to use LLaMA 2, False to use LLaMA 3.1
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Generate rhyming words for nonsense words.')
    # parser.add_argument('--input_file', type=str, default='dataset/rhyming_test.csv', help='Input file with nonsense words.')
    parser.add_argument('--LLM', type=str, choices=['yi', 'llama3', 'mistral', 'falcon', 'llama3.1', 'qwen', 'lora-llama3.1'], default='llama3.1', help='Choose between LLaMA 2, LLaMA 3, and Mistral.')
    parser.add_argument('--prompt_template', type=str, choices=['simple2', 'icl', 'simple', 'split'], default='simple', help='Choose between COT and ICL.')
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
        cache_dir = "../LLM/Yi-1.5-6B-Chat"
    elif args.LLM == 'falcon':
        model_id = "tiiuae/Falcon3-7B-Instruct"
    elif args.LLM == 'gemma':
        model_id = "google/gemma-7b-it"
    elif args.LLM == 'qwen':
        model_id = "Qwen/Qwen2.5-7B-Instruct"
    elif args.LLM == 'lora-llama3.1':
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        lora_model_path = "model/meta-llama/Meta-Llama-3.1-8B-Instruct_IPA_model"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        IPA_TOKENS = ['<IPA>', '</IPA>']
        special_tokens = {"additional_special_tokens": IPA_TOKENS}
        tokenizer.add_special_tokens(special_tokens) 
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        use_direct_inference = True
    else:
        raise ValueError("Invalid model choice. Choose from 'llama2', 'llama3', or 'mistral'.")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if args.LLM != "lora-llama3.1":
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

    LLM = args.LLM
    if args.LLM == 'lora-llama3.1':
        LLM = 'llama3.1'
    input_file_good = f'datasets/arpabet_data_{LLM}_good.csv'
    input_file_bad = f'datasets/arpabet_data_{LLM}_bad.csv'
    if args.prompt_template == 'simple':
        prompt_template = prompt_syllables_word_simple

    input_df_good = pd.read_csv(input_file_good)
    input_df_bad = pd.read_csv(input_file_bad)
    use_lora = args.LLM == 'lora-llama3.1'
    # Prepare prompts
    single_prompts = []
    multi_prompts = []
    for idx, row in input_df_good.iterrows():
        word = row['word']
        prompt_single = apply_template(word, prompt_template, tokenizer, lora=use_lora)
        single_prompts.append(prompt_single)
    for idx, row in input_df_bad.iterrows():
        word = row['word']
        prompt_single = apply_template(word, prompt_template, tokenizer, lora=use_lora)
        multi_prompts.append(prompt_single)

    if args.LLM == 'lora-llama3.1':
        labels_single = []
        for prompt in tqdm(single_prompts, desc="Processing good words"):
            response = generate_response(prompt, model, tokenizer, device, max_new_tokens=256)
            single_syllables = extract_syllables(response)
            print(response, single_syllables)
            labels_single.append(single_syllables)
            
        labels_multi = []
        for prompt in tqdm(multi_prompts, desc="Processing bad words"):
            response = generate_response(prompt, model, tokenizer, device, max_new_tokens=256)
            multi_syllables = extract_syllables(response)
            print(response, multi_syllables)
            labels_multi.append(multi_syllables)
    else:
        single_responses = llm.generate(single_prompts, sampling_params=sampling_params)
        multi_responses = llm.generate(multi_prompts, sampling_params=sampling_params)

        labels_single = []
        labels_multi = []
        for single_response, multi_response in zip(single_responses, multi_responses):
            single_response = single_response.outputs[0].text
            multi_response = multi_response.outputs[0].text
            single_syllables = extract_syllables(single_response)
            multi_syllables = extract_syllables(multi_response)
            print(single_response, single_syllables)
            labels_single.append(single_syllables)
            labels_multi.append(multi_syllables)
    
    # single_prompts_2 = []
    # multi_prompts_2 = []
    # for single_response, multi_response in zip(single_responses, multi_responses):
    #     single_response = single_response.outputs[0].text
    #     multi_response = multi_response.outputs[0].text
    #     single_response_cleaned = single_response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
    #     multi_response_cleaned = multi_response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
    #     single_prompt_2 = extract_answer_template(single_response_cleaned, tokenizer)
    #     multi_prompt_2 = extract_answer_template(multi_response_cleaned, tokenizer)
    #     single_prompts_2.append(single_prompt_2)
    #     multi_prompts_2.append(multi_prompt_2)
    
    # single_responses_2 = llm.generate(single_prompts_2, sampling_params=sampling_params)
    # multi_responses_2 = llm.generate(multi_prompts_2, sampling_params=sampling_params)

    # labels_single = []
    # labels_multi = []
    # for single_response_2, multi_response_2 in zip(single_responses_2, multi_responses_2):
    #     single_response_2 = single_response_2.outputs[0].text
    #     multi_response_2 = multi_response_2.outputs[0].text
    #     single_response_2_cleaned = single_response_2.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
    #     multi_response_2_cleaned = multi_response_2.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
    #     labels_single.append(single_response_2_cleaned)
    #     labels_multi.append(multi_response_2_cleaned)
    
    input_df_good['prediction'] = labels_single
    input_df_bad['prediction'] = labels_multi
    output_file = f"results/syllables_{args.prompt_template}_{args.LLM}_good.csv"
    input_df_good.to_csv(output_file, index=False)
    output_file = f"results/syllables_{args.prompt_template}_{args.LLM}_bad.csv"
    input_df_bad.to_csv(output_file, index=False)
    
    
    
    
    


