import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, T5EncoderModel, AutoModel
import pandas as pd
import os
import pickle
import numpy as np

# Check if a GPU is available
access_token ='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'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
print(f"Using device: {device}")
def init_model(LLM, device):
    """
    Initialize the model and tokenizer, load them on the specified device.
    """
    if LLM == 'llama3.1':
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model_id = "/model-weights/Meta-Llama-3.1-8B-Instruct"
    elif LLM == 'llama3':
        model_id = "meta-llama/Meta-Llama-3-8B"
    elif LLM == 'llama2':
        model_id = "meta-llama/Llama-2-7b-hf"
    elif LLM == 'bert':
        model_id = "bert-base-uncased"
    elif LLM == 'bloom':
        model_id = "bigscience/bloom-560m"
    elif LLM == 'gpt-neo':
        model_id = "EleutherAI/gpt-neo-2.7B"
    elif LLM == 'gemma-2b':
        model_id = "google/gemma-1.1-2b-it"
    elif LLM == 'mistral':
        model_id = "/model-weights/Mistral-7B-Instruct-v0.3"
    elif LLM == 'gpt2':
        model_id = "gpt2"
    elif LLM == 'ByT5-small':
        model_id = "google/byt5-small"
    elif LLM == 'ByT5':
        model_id = "google/byt5-base"
    elif LLM == 'mT5':
        model_id = "google/mt5-base"
    elif LLM == 'qwen-0.5b':
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    elif LLM == 'qwen-1.5b':
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    elif LLM == 'llama3.2-1b':
        model_id = "meta-llama/Meta-Llama-3.2-1B-Instruct"
    elif LLM == 'yi':
        model_id = "01-ai/Yi-1.5-6B-Chat"
        model_dir = "../LLM/Yi-1.5-6B-Chat"

    else:
        raise ValueError(f"Model {LLM} not supported")
    
    if LLM in ['ByT5', 'mT5', 'ByT5-small']:
        model = T5EncoderModel.from_pretrained(model_id, token=access_token).to(device)
    elif LLM == 'bert':
        model = AutoModel.from_pretrained(model_id, token=access_token).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token).to(device)
    
    if LLM in ['ByT5', 'mT5']:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    return tokenizer, model

def extract_embeddings_for_all_layers(model, tokenizer, prompt, device):
    """
    Generate embeddings for all layers for the last token of the prompt, using the GPU if available.
    
    Args:
        model: LLaMA model instance.
        tokenizer: Tokenizer for the LLaMA model.
        prompt: The concatenated string of word1 and word2.
        device: The device to run the model on (GPU or CPU).

    Returns:
        A dictionary where keys are layer indices and values are embeddings of the last token.
    """
    # Tokenize the prompt and convert to tensors, then move them to the correct device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        if isinstance(model, T5EncoderModel):
            # For T5-based models (ByT5, mT5)
            outputs = model(input_ids=inputs['input_ids'], 
                          attention_mask=inputs['attention_mask'],
                          output_hidden_states=True)
            # Get hidden states from encoder
            hidden_states = outputs.hidden_states
        else:
            # For causal language models (LLaMA, GPT-2)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
    
    # Dictionary to store embeddings for each layer
    embeddings_by_layer = {}

    # Iterate over all layers and get the embedding for the last token
    for layer_num, hidden_states in enumerate(outputs.hidden_states):
        # Extract the embedding for the last token of the sequence
        last_token_embedding = hidden_states[:, -1, :].squeeze(0).cpu().numpy()  # Move to CPU before converting to numpy
        embeddings_by_layer[layer_num] = last_token_embedding

    return embeddings_by_layer

import pickle

def process_csv_ipa(file_path, LLM, layers, prefix, feature):
    df = pd.read_csv(file_path)
    tokenizer, model = init_model(LLM, device)

    if layers == [-1]:
        num_hidden_layers = model.config.num_hidden_layers
        layers = list(range(num_hidden_layers + 1))

    dataset_name = os.path.basename(file_path).replace('.csv', '')
    if prefix:
        output_dir = f"datasets/{prefix}_{dataset_name}_{LLM}_{feature}_IPA_layer_embeddings_pkl"
    else:
        output_dir = f"datasets/none_{dataset_name}_{LLM}_{feature}_IPA_layer_embeddings_pkl"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries to hold embeddings per layer
    embeddings_per_layer = {layer_num: [] for layer_num in layers}
    ipas_list = []
    feature_vec_list = []

    # Iterate over each row in the CSV file
    for idx, row in df.iterrows():
        ipa = row['ipa']
        feature_vec = row[feature]  # Get the specific feature column
        # Convert feature value to numeric: '+' -> 1, '-' -> -1, '0' -> 0
        if feature_vec == '+':
            feature_vec = 1
        elif feature_vec == '-':
            feature_vec = -1
        elif feature_vec == '0':
            feature_vec = 0

        if prefix:
            prompt = f"{prefix} /{ipa}/"
        else:
            prompt = f"/{ipa}/"

        embeddings_by_layers = extract_embeddings_for_all_layers(model, tokenizer, prompt, device)

        ipas_list.append(ipa)
        feature_vec_list.append(feature_vec)

        for layer_num in layers:
            embedding = embeddings_by_layers[layer_num]
            embeddings_per_layer[layer_num].append(embedding)

    # After processing all rows, save embeddings per layer
    for layer_num in layers:
        layer_data = {
            'embeddings': embeddings_per_layer[layer_num],
            'ipas': ipas_list,
            'features': feature_vec_list
        }
        
        output_file = os.path.join(output_dir, f'layer_{layer_num}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(layer_data, f)

def process_csv_reg(file_path, LLM, layers, prefix, use_IPA, use_slash):
    df = pd.read_csv(file_path)
    tokenizer, model = init_model(LLM, device)

    if layers == [-1]:
        num_hidden_layers = model.config.num_hidden_layers
        layers = list(range(num_hidden_layers + 1))

    if use_IPA:
        output_type = "IPA"
    elif use_slash:
        output_type = "slash"
    else:
        output_type = "word"

    dataset_name = os.path.basename(file_path).replace('.csv', '')
    if prefix:
        output_dir = f"embeddings/{prefix}_{dataset_name}_{LLM}_{output_type}_layer_embeddings"
    else:
        output_dir = f"embeddings/none_{dataset_name}_{LLM}_{output_type}_layer_embeddings"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries to hold embeddings per layer
    embeddings_per_layer = {layer_num: [] for layer_num in layers}
    words_list = []
    phon_vec_list = []
    syllable_count_list = []

    # Iterate over each row in the CSV file
    for idx, row in df.iterrows():
        word = row['word']
        feature = eval(row['phon_vec'])
        syllable_count = len(eval(row['syllables']))

        if prefix:
            prompt = f"{prefix} {word}"
        else:
            prompt = f"{word}"

        embeddings_by_layers = extract_embeddings_for_all_layers(model, tokenizer, prompt, device)

        words_list.append(word)
        phon_vec_list.append(feature)  # Convert string representation to list
        syllable_count_list.append(syllable_count)

        for layer_num in layers:
            embedding = embeddings_by_layers[layer_num]
            embeddings_per_layer[layer_num].append(embedding)

    # After processing all rows, save embeddings per layer
    for layer_num in layers:
        layer_file_path = os.path.join(
            output_dir,
            f"layer_{layer_num}.pkl"
        )
        # Prepare data to pickle
        data_to_save = {
            'word': words_list,
            'embeddings': embeddings_per_layer[layer_num],
            'phon_vecs': phon_vec_list,
            'syllable_count': syllable_count_list
        }
        # Save to pickle file
        with open(layer_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)


                    
def process_csv(file_dir, file_name, LLM, layers, prefix, use_IPA, use_slash):
    """
    Process the CSV file, generate prompts, extract embeddings for specific layers,
    and save embeddings for each layer into separate CSV files.
    
    Args:
        file_path: Path to the input CSV file.
        LLM: The name of the LLaMA model to use (e.g., llama2 or llama3).
        layers: The layers from which to extract embeddings. If layers == -1, use all available layers.
        prefix: Optional prefix to add to the prompt and output directory name.
    """
    df = pd.read_csv(os.path.join(file_dir, file_name))
    tokenizer, model = init_model(LLM, device)

    # Determine the number of layers if layers == -1
    if layers == [-1]:
        # Get the total number of layers from the model's config
        num_hidden_layers = model.config.num_hidden_layers
        layers = list(range(num_hidden_layers+1))  # Generate a list of all layers [0, 1, ..., num_hidden_layers-1]

    # Determine whether to use IPA, slash, or word in file names and output directories
    if use_IPA:
        output_type = "IPA"
    elif use_slash:
        output_type = "slash"
    else:
        output_type = "word"

    # Modify output directories to include model name, dataset name, and prefix
    dataset_name = os.path.basename(file_name).replace('.csv', '')
    if prefix:
        output_dir = f"embeddings/{prefix}_{dataset_name}_{LLM}_{output_type}_layer_embeddings"
    else:
        output_dir = f"embeddings/none_{dataset_name}_{LLM}_{output_type}_layer_embeddings"
    
    os.makedirs(output_dir, exist_ok=True)
    embeddings_per_layer1 = {}
    embeddings_per_layer2 = {}
    labels = []
    # Iterate over each row in the CSV file
    for idx, row in df.iterrows():
        word1 = row['word1']
        word2 = row['word2']
        if use_IPA:
            word1 = row['word1_IPA']
            word2 = row['word2_IPA']
        elif use_slash:
            word1 = '/'.join(list(word1))
            word2 = '/'.join(list(word2))
        label = row['label']
        labels.append(label)
        
        # Create the prompt based on whether a prefix is provided
        if prefix:
            prompt1 = f"{prefix} {word1} {word2}"
        else:
            prompt1 = f"{word1} {word2}"
        
        # Extract embeddings for all layers
        embeddings_by_layers1 = extract_embeddings_for_all_layers(model, tokenizer, prompt1, device)

        for layer_num, embedding in embeddings_by_layers1.items():
            if layer_num in embeddings_per_layer1:
                embeddings_per_layer1[layer_num].append(embedding)
            else:
                embeddings_per_layer1[layer_num] = [embedding]


        
    # Save each layer's embedding in a separate pickle file
    for layer_num, embedding in embeddings_per_layer1.items():
        if layer_num in layers:
            layer_file_path = os.path.join(output_dir, f"layer_{layer_num}.pkl")
            
            # Prepare data to pickle
            data_to_save = {
                'word1': word1,
                'word2': word2,
                'embedding': np.array(embedding),
                'label': labels
            }
            
            # Save to pickle file
            with open(layer_file_path, 'wb') as f:
                pickle.dump(data_to_save, f)

def main():
    parser = argparse.ArgumentParser(description="Process CSV and extract embeddings from LLaMA models.")
    
    # Add command line arguments for file_path, LLM, layers, and optional prefix
    parser.add_argument('--file_dir', type=str, required=True, help="directory of the csv files.")
    parser.add_argument('--file_name', type=str, required=True, help="name of the csv file.")
    parser.add_argument('--LLM', type=str, required=True, help="Model name to use")
    parser.add_argument('--layers', type=int, nargs='+', required=True, help="List of layer indices to extract embeddings from. Use -1 for all layers.")
    parser.add_argument('--prefix', type=str, default='', help="Optional prefix to prepend to the prompt and output directory. Default is no prefix.")
    parser.add_argument('--use_IPA', action='store_true', help="use the IPA to do inference")
    parser.add_argument('--use_slash', action='store_true', help="use slash to do inference")
    parser.add_argument('--feature', type=str, default='rhyme', choices=['rhyme', 'g2p'], help="The feature to use for the inference")
    args = parser.parse_args()
    
    # Call the processing function with the user inputs
    if args.feature == 'rhyme':
        process_csv(args.file_dir, args.file_name, args.LLM, args.layers, args.prefix, args.use_IPA, args.use_slash)
    elif args.feature == 'g2p':
        file_path = os.path.join(args.file_dir, args.file_name)
        process_csv_reg(file_path, args.LLM, args.layers, args.prefix, args.use_IPA, args.use_slash)

if __name__ == "__main__":
    main()
