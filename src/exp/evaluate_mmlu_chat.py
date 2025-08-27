import os
import argparse
import re
import torch
import pandas as pd
from tqdm import tqdm
from thefuzz import process
import random
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

'''
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

pip install thefuzz
python eval/evaluate_chat_mmlu.py -d data/mmlu/data/
'''

def load_vllm(model_name_or_path):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return llm, tokenizer

def load_models_tokenizer(args):
    if args.use_vllm:
        return load_vllm(args.checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    
    # Handle special tokens if needed
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    if args.use_lora:
        # Load the base model with appropriate dtype
        IPA_TOKENS = ['<IPA>', '</IPA>']
        special_tokens = {"additional_special_tokens": IPA_TOKENS}
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        
        # Load the LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if args.load:
            print(f"Loading model weights from {args.load}")
            model_state = torch.load(args.load, map_location="cpu")
            model.load_state_dict(model_state, strict=False)
            model.half().cuda()
    
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    return model, tokenizer


def format_example(line):
    example = (
        "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question. Give the answer directly.\n\n"
        + line["question"]
        + "\n"
    )
    for choice in choices:
        example += f'{choice}. {line[f"{choice}"]}\n'
    return example


def process_before_extraction(gen, choice_dict):
    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen


def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(
        response, {choice: row[choice] for choice in choices}
    )
    pred = extract_choice(gen, [row[choice] for choice in choices])
    return pred


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    save_result_dir=None,
    overwrite=False,
    use_vllm=False,
    **kwargs
):
    result_path = os.path.join(save_result_dir, f"{subject_name}_result.csv")
    if not overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!")
        score = []
        for (_, datarow), (_, resultrow) in zip(
            test_df.iterrows(), pd.read_csv(result_path).astype(str).iterrows()
        ):
            # pred = extract_answer(resultrow['model_response'], datarow)
            pred = resultrow["model_output"]
            correct = 1 if pred == datarow["answer"] else 0
            score.append(correct)
        return score

    result = []
    score = []
    responses = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row)
        
        if use_vllm:
            # Format for vLLM
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
            chat_text = tokenizer.apply_chat_template(chat, tokenize=False)
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
            outputs = model.generate(chat_text, sampling_params)
            response = outputs[0].outputs[0].text
        else:
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
            chat_text = tokenizer.apply_chat_template(chat, tokenize=False)
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
                temperature=0.0
            )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print(question)
        print(response)
        pred = extract_answer(response, row)
        print(pred)
        print("======================")

        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            if args.debug:
                print(f'{question} pred: {pred} ref: {row["answer"]}')
        result.append(pred)
        responses.append(response)

    if save_result_dir:
        test_df["model_output"] = result
        test_df["model_response"] = responses
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return score


def cal_mmlu(res, save_result_dir):
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0
    
    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.0
        acc_norm_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    print("\n\n\n")
    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            print("%s ACC: %.2f " % (k, acc_sum_dict[k] * 100 / cnt_dict[k]))
    print("AVERAGE ACC:%.2f " % (acc_sum * 100 / cnt))
    with open(os.path.join(save_result_dir, "mmlu_result.txt"), "w") as f:
        f.write(f"AVERAGE ACC: {acc_sum * 100 / cnt}\n")
        for k in TASK_NAME_MAPPING.keys():
            if k in cnt_dict:
                f.write(f"{k} ACC: {acc_sum_dict[k] * 100 / cnt_dict[k]}\n")


def main(args):
    print("loading model weights")
    if args.checkpoint_path is not None:
        model, tokenizer = load_models_tokenizer(args)
    else:
        model, tokenizer = None, None
    print("model loaded")

    # Create a more descriptive output directory if LoRA is used
    if args.use_lora:
        lora_name = os.path.basename(args.lora_path)
        args.output_dir = os.path.join(args.output_dir, f"lora_{lora_name}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.csv"
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        ).astype(str)

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            test_df,
            save_result_dir=args.output_dir,
            overwrite=args.overwrite,
            use_vllm=args.use_vllm,
        )
        dev_result[subject_name] = score
    cal_mmlu(dev_result, args.output_dir)


TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
ALL_SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
random.seed(1234)
# SUBJECTS = random.sample(SUBJECTS, 1)
# three subjects from each category
SUBJECTS = []
for k, v in TASK_NAME_MAPPING.items():
    subjects = TASK_NAME_MAPPING[k]
    SUBJECTS.extend(random.sample(subjects, 3))
new_TASK_NAME_MAPPING = {}
for k, v in TASK_NAME_MAPPING.items():
    for vv in v:
        if vv in SUBJECTS:
            if k not in new_TASK_NAME_MAPPING:
                new_TASK_NAME_MAPPING[k] = []
            new_TASK_NAME_MAPPING[k].append(vv)
TASK_NAME_MAPPING = new_TASK_NAME_MAPPING
print(TASK_NAME_MAPPING)
choices = ["A", "B", "C", "D"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, help="Path to eval data")
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existed results",
    )
    
    # Add new arguments for LoRA and vLLM
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapter")
    parser.add_argument("--lora_path", type=str, default='model/meta-llama/Meta-Llama-3.1-8B-Instruct_IPA_model', help="Path to LoRA adapter")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for inference")
    parser.add_argument("--load", type=str, default=None, help="Load quantized model")
    parser.add_argument("--output_dir", type=str, default="outs_chat/mmlu_eval_result", 
                        help="Directory to save results")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 precision")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Use flash attention")

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)