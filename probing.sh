# generate embedding
python probing/generate_embedding.py --file_dir datasets --file_name arpabet_data_llama3_good.csv --LLM llama3.1 --layers -1 --feature g2p
python probing/generate_embedding.py --file_dir datasets --file_name arpabet_data_llama3_bad.csv --LLM llama3.1 --layers -1 --feature g2p

# train probe
python probing/train_probe_g2p.py --LLM llama3.1
python probing/train_probe_syl.py --LLM llama3.1 

# control task
python probing/train_probe_g2p.py --LLM llama3.1 --control_task_label
python probing/train_probe_syl.py --LLM llama3.1 --control_task_label