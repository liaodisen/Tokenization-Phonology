MODEL_NAME_OR_PATH=meta-llama/Meta-Llama-3-8B-Instruct
DATA_PATH_PHONETIC=./dataset/rhyming_dataset_phon.jsonl
MODEL_MAX_LENGTH=2048
BATCH_SIZE=2
OUTPUT=./output
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 ./src/finetune.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path_phonetic $DATA_PATH_PHONETIC \
    --deepspeed ./src/ds_config_zero2.json \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate 2e-3 \
    --logging_steps 5 \
    --output_dir $OUTPUT \
