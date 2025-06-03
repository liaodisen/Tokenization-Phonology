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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
import json
from scipy.stats import ttest_rel

def load_embeddings_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    word = data['word']
    embeddings = data['embeddings']
    phon_vec = data['phon_vecs']
    return word, embeddings, phon_vec

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
    elif llm in ["bloom", "pythia"]:
        layers = 24
    elif llm == "bert":
        layers = 12
    elif llm == "falcon":
        layers = 18

    results = {}

    for i in range(layers):
        r2_scores_bad = []
        r2_scores_good = []
        for seed in range(10, 20):
            np.random.seed(seed)
            torch.manual_seed(seed)

            fp1 = f'embeddings/{args.prefix}_arpabet_data_{args.LLM}_bad_{args.LLM}_word_layer_embeddings/layer_{i}.pkl'
            fp2 = f'embeddings/{args.prefix}_arpabet_data_{args.LLM}_good_{args.LLM}_word_layer_embeddings/layer_{i}.pkl'
            word, embeddings_bad, phon_vec_bad = load_embeddings_pkl(fp1)
            word_good, embeddings_good, phon_vec_good = load_embeddings_pkl(fp2)
            label_bad, label_good = np.array(phon_vec_bad), np.array(phon_vec_good)
            if args.control_task_label:
                label_bad = np.random.randint(0, 40, size=label_bad.shape)
                label_good = np.random.randint(0, 40, size=label_good.shape)
            elif args.control_task_embed:
                embeddings_bad = np.random.randn(*np.array(embeddings_bad).shape)
                embeddings_good = np.random.randn(*np.array(embeddings_good).shape)
            y_bad = label_bad[:, :]
            y_good = label_good[:, :]
            y_bad = (y_bad - y_bad.mean(axis=0)) / y_bad.std(axis=0)
            y_good = (y_good - y_good.mean(axis=0)) / y_good.std(axis=0)
            X_bad = np.array(embeddings_bad)
            X_good = np.array(embeddings_good)
            X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(X_bad, y_bad, test_size=0.2, random_state=seed)
            X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(X_good, y_good, test_size=0.2, random_state=seed)
            model = RidgeCV(alphas=[10, 100, 500, 1000, 2000])
            model.fit(X_train_bad, y_train_bad)
            model_good = RidgeCV(alphas=[10, 100, 500, 1000, 2000])
            model_good.fit(X_train_good, y_train_good)

            y_pred_bad = model.predict(X_test_bad)
            y_pred_good = model_good.predict(X_test_good)

            r2_bad = r2_score(y_test_bad, y_pred_bad)
            r2_good = r2_score(y_test_good, y_pred_good)
            print(f"Layer {i} - R2 Bad: {r2_bad:.4f}")
            print(f"Layer {i} - R2 Good: {r2_good:.4f}")
            r2_scores_bad.append(r2_bad)
            r2_scores_good.append(r2_good)


        # Calculate one-tailed p-value using paired t-test
        t_stat, p_value = ttest_rel(r2_scores_good, r2_scores_bad)
        p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)

        results[f"layer_{i}"] = {
            "r2_scores_bad": r2_scores_bad,
            "r2_scores_good": r2_scores_good,
            "mean_bad": np.mean(r2_scores_bad),
            "mean_good": np.mean(r2_scores_good),
            "p_value_one_tailed": p_value_one_tailed,
            "std_bad": np.std(r2_scores_bad),
            "std_good": np.std(r2_scores_good),
        }

    # Save results to a JSON file
    if args.control_task_label:
        save_path = f'results/probing_results_g2p_{args.LLM}_{args.prefix}_control_task_label.json'
    elif args.control_task_embed:
        save_path = f'results/probing_results_g2p_{args.LLM}_{args.prefix}_control_task_embed.json'
    else:
        save_path = f'results/probing_results_g2p_{args.LLM}_{args.prefix}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
