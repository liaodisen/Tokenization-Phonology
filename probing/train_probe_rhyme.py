import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
import json

def load_embeddings_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    word1 = data['word1']
    word2 = data['word2']
    embeddings = data['embedding']
    label = data['label']
    return word1, word2, embeddings, label

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=700, num_classes_per_output=3):
        super(MLPClassifier, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Single output layer with output_size x num_classes_per_output
        self.output_layer = nn.Linear(hidden_size, num_classes_per_output)

    def forward(self, x):
        x = self.hidden(x)
        # Reshape to [batch_size, output_size, num_classes_per_output]
        x = self.output_layer(x)
        return x


def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for embeddings, phon_vecs in tqdm(train_loader, desc="Training", leave=False):
        embeddings = embeddings.to(device)
        phon_vecs = phon_vecs.to(device)
        
        # Map phon_vecs values from {-1, 0, 1} to {0, 1, 2}
        phon_vecs = phon_vecs + 1  # Now phon_vecs will be in the range {0, 1, 2}
        
        optimizer.zero_grad()
        outputs = model(embeddings)  # Shape: [batch_size, 3]
        
        # Compute loss
        loss = criterion(outputs, phon_vecs.view(-1))  # No need to flatten outputs
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    return average_loss
def evaluate_model(model, data_loader, device):
    model.eval()
    total_acc = 0
    total_words = 0
    with torch.no_grad():
        for embeddings, phon_vecs in tqdm(data_loader, desc="Evaluating", leave=False):
            embeddings = embeddings.to(device)
            phon_vecs = phon_vecs.to(device)
            
            # Map phon_vecs values from {-1, 0, 1} to {0, 1, 2}
            phon_vecs = phon_vecs + 1
            
            outputs = model(embeddings)  # Shape: [batch_size, 3]
            preds = outputs.argmax(dim=1)  # Shape: [batch_size]
            
            # Calculate accuracy
            batch_acc = (preds == phon_vecs.view(-1)).float().mean().item()
            total_acc += batch_acc
            total_words += 1

    average_acc = total_acc / total_words if total_words > 0 else 0
    return average_acc


def main():
    parser = argparse.ArgumentParser(description="Train model on embeddings to predict rhyming.")
    parser.add_argument("--LLM", type=str, default="gpt2", help="LLM to use for training")
    parser.add_argument("--prefix", type=str, default="none", help="Prefix to use for training")
    parser.add_argument('--control_task_label', action='store_true', help="whether to randomize the label as the control task.")
    parser.add_argument('--control_task_embed', action='store_true', help="whether to randomize the embedding as the control task.")

    args = parser.parse_args()

    # Load dataset
    llm = args.LLM
    if llm in ["gpt2", "ByT5-small"]:
        layers = 13
    elif llm in ["llama3", "mistral", "llama2", "yi", "gpt-neo", "llama3.1"]:
        layers = 32
    elif llm in ["gemma", "qwen", "qwen2"]:
        layers = 28
    elif llm in ["bloom", "pythia", "qwen-0.5b"]:
        layers = 24
    elif llm == "bert":
        layers = 12
    elif llm in ["falcon", "ByT5"]:
        layers = 18

    results = {}

    for i in range(layers):
        accuracies_slash = []
        accuracies_word = []

        for seed in range(10):
            np.random.seed(seed)
            torch.manual_seed(seed)

            fp1 = f'embeddings/{args.prefix}_rhyming_pairs_{args.LLM}_slash_layer_embeddings/layer_{i}.pkl'
            fp2 = f'embeddings/{args.prefix}_rhyming_pairs_{args.LLM}_word_layer_embeddings/layer_{i}.pkl'
            word1, word2, embeddings_slash, label_slash = load_embeddings_pkl(fp1)
            word1_good, word2_good, embeddings_word, label_word = load_embeddings_pkl(fp2)
            label_slash, label_word = np.array(label_slash), np.array(label_word)
            if args.control_task_label:
                label_slash = np.random.randint(0, 2, size=len(label_slash))
                label_word = np.random.randint(0, 2, size=len(label_word))
            elif args.control_task_embed:
                embeddings_slash = np.random.randn(*embeddings_slash.shape)
                embeddings_word = np.random.randn(*embeddings_word.shape)
            y_slash = label_slash
            y_word = label_word
            X_slash = np.array(embeddings_slash)
            X_word = np.array(embeddings_word)
            X_train_slash, X_test_slash, y_train_slash, y_test_slash = train_test_split(X_slash, y_slash, test_size=0.2, random_state=seed)
            X_train_word, X_test_word, y_train_word, y_test_word = train_test_split(X_word, y_word, test_size=0.2, random_state=seed)
            model_slash = LogisticRegression(max_iter=1000, C=10)
            model_slash.fit(X_train_slash, y_train_slash)
            model_word = LogisticRegression(max_iter=1000, C=10)
            model_word.fit(X_train_word, y_train_word)

            y_pred_slash = model_slash.predict(X_test_slash)
            y_pred_word = model_word.predict(X_test_word)

            accuracy_slash = accuracy_score(y_test_slash, y_pred_slash)
            accuracy_word = accuracy_score(y_test_word, y_pred_word)

            accuracies_slash.append(accuracy_slash)
            accuracies_word.append(accuracy_word)

        # Calculate p-value
            t_stat, p_value_slash = ttest_rel(accuracies_slash, accuracies_word)
            p_value_one_tailed = p_value_slash / 2 if t_stat > 0 else 1 - (p_value_slash / 2)

        results[f"layer_{i}"] = {
            "accuracies_slash": accuracies_slash,   
            "accuracies_word": accuracies_word,
            "std_slash": np.std(accuracies_slash),
            "std_word": np.std(accuracies_word),
            "mean_slash": np.mean(accuracies_slash),
            "mean_word": np.mean(accuracies_word),
            "p_value_one_tailed": p_value_one_tailed,
        }

        print(f"Layer {i} - Mean Accuracy Slash: {np.mean(accuracies_slash):.4f}")
        print(f"Layer {i} - Mean Accuracy Word: {np.mean(accuracies_word):.4f}")
        print(f"Layer {i} - One-tailed P-Value Slash: {p_value_one_tailed:.10f}")

    # Save results to a JSON file
    if args.control_task_label:
        save_path = f'results/probing_results_rhyme_{args.LLM}_{args.prefix}_control_task_label'
    elif args.control_task_embed:
        save_path = f'results/probing_results_rhyme_{args.LLM}_{args.prefix}_control_task_embed'
    else:
        save_path = f'results/probing_results_rhyme_{args.LLM}_{args.prefix}'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
