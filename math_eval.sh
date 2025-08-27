#! /bin/bash
#SBATCH --job-name=math-eval-job          # Job name
#SBATCH --output=output/math-eval-job.out        # Output file
#SBATCH --error=output/math-eval-job.err         # Error file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --time=16:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=64GB

MODEL=model/meta-llama/Meta-Llama-3.1-8B-Instruct_IPA_model
device=0

# evaluate with lora
CUDA_VISIBLE_DEVICES=$device python src/exp/evaluate_math.py \
    --model llama3.1 \
    --data_root ./finetune_datasets \
    --use_lora \
    --lora_path ./model/meta-llama/Meta-Llama-3.1-8B-Instruct_IPA_model

# no lora
CUDA_VISIBLE_DEVICES=$device python src/exp/evaluate_math.py \
    --model llama3.1 \
    --data_root ./finetune_datasets