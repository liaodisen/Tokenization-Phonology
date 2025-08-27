#!/bin/bash
#SBATCH --job-name=finetune2-job          # Job name
#SBATCH --output=output/finetune2-job.out        # Output file
#SBATCH --error=output/finetune2-job.err         # Error file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --time=5:00:00                  # Time limit hrs:min:sec
#SBATCH --mem=64GB


MODEL_NAME_OR_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
DATA_PATH_PHONETIC=finetune_datasets/phon_processed.jsonl
DATA_PATH_CONVERSATION=finetune_datasets/conversation_processed_new.jsonl
MODEL_MAX_LENGTH=2048
BATCH_SIZE=2
OUTPUT=output
export CUDA_VISIBLE_DEVICES=0,1
# # single GPU
# python ./src/finetune2.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --data_path_phonetic $DATA_PATH_PHONETIC \
#     --data_path_conversation $DATA_PATH_CONVERSATION \
#     --bf16 True \
#     --optim "adamw_8bit" \
#     --model_max_length $MODEL_MAX_LENGTH \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --per_device_eval_batch_size $BATCH_SIZE \
#     --learning_rate 2e-4 \
#     --evaluation_strategy no \
#     --logging_steps 5 \
#     --local_rank -1 \
#     --gradient_accumulation_steps 1 \
#     --report_to wandb \
#     --output_dir $OUTPUT \
#     --use_lora True \
#     --save_strategy no \

# multi GPU
torchrun --nproc_per_node=2 ./src/finetune2.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path_phonetic $DATA_PATH_PHONETIC \
    --data_path_conversation $DATA_PATH_CONVERSATION \
    --bf16 True \
    --optim "adamw_8bit" \
    --model_max_length $MODEL_MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate 2e-4 \
    --evaluation_strategy no \
    --logging_steps 5 \
    --local_rank -1 \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --output_dir $OUTPUT \
    --use_lora True \
    --save_strategy no \
    --ddp_find_unused_parameters False \