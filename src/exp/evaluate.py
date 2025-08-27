import pandas as pd
import numpy as np
import editdistance

def calculate_per(gt, pred):
    if pd.isna(pred):
        return 0.5
    else:
        distance = editdistance.eval(gt, pred)
        per = distance / len(gt)
        return min(per, 1)

def average_per(gt, pred):
    per_list = [calculate_per(gt[i], pred[i]) for i in range(len(gt))]

    return sum(per_list) / len(per_list)


def evaluate_rhyme_prediction(rhyme_prediction_file: str):
    rhyme_prediction_df = pd.read_csv(rhyme_prediction_file)
    true_rhymes = rhyme_prediction_df["label"]
    predicted_rhymes = rhyme_prediction_df["prediction"].astype(str)
    predicted_rhymes = predicted_rhymes.apply(lambda x: True if 'True' in x else False if 'False' in x else None)
    accuracy = (true_rhymes == predicted_rhymes).mean()
    recall = (true_rhymes & predicted_rhymes).mean()
    precision = (true_rhymes & predicted_rhymes).mean() / predicted_rhymes.mean()
    return accuracy, recall, precision


def evaluate_rhyme2(rhyme_prediction_file: str):
    rhyme_prediction_df = pd.read_csv(rhyme_prediction_file)
    true_rhymes = rhyme_prediction_df["label"]
    predicted_rhymes_single = rhyme_prediction_df["single_label"]
    predicted_rhymes_multi = rhyme_prediction_df["multi_label"]
    predicted_rhymes_single = predicted_rhymes_single.apply(lambda x: True if 'Yes' in x else False if 'No' in x else None)
    predicted_rhymes_multi = predicted_rhymes_multi.apply(lambda x: True if 'Yes' in x else False if 'No' in x else None)
    accuracy_single = (true_rhymes == predicted_rhymes_single).mean()
    accuracy_multi = (true_rhymes == predicted_rhymes_multi).mean()
    return accuracy_single, accuracy_multi

def evaluate_syllable_prediction(syllable_prediction_file: str):
    syllable_prediction_df = pd.read_csv(syllable_prediction_file)
    true_syllables = syllable_prediction_df["syllables"]
    true_num_syllables = syllable_prediction_df["num_syl"]
    predicted_num_syllables = syllable_prediction_df["prediction"]
    accuracy = (true_num_syllables == predicted_num_syllables).mean()
    return accuracy


def evaluate_arpabet_prediction(arpabet_prediction_file: str):
    arpabet_prediction_df = pd.read_csv(arpabet_prediction_file)
    true_arpabet = arpabet_prediction_df["arpabet"].apply(lambda x: " ".join(eval(x)))
    predicted_arpabet = arpabet_prediction_df["prediction"]
    per = average_per(true_arpabet, predicted_arpabet)
    return per


if __name__ == "__main__":
    LLM = "mistral"
    acc_syl1 = evaluate_syllable_prediction(f'results/syllables_simple_{LLM}_bad.csv')
    acc_syl2 = evaluate_syllable_prediction(f'results/syllables_simple_{LLM}_good.csv')
    # acc_syl1_lora = evaluate_syllable_prediction(f'results/syllables_simple_lora-{LLM}_bad.csv')
    # acc_syl2_lora = evaluate_syllable_prediction(f'results/syllables_simple_lora-{LLM}_good.csv')
    per1 = evaluate_arpabet_prediction(f'results/arpabet_simple_{LLM}_bad.csv')
    per2 = evaluate_arpabet_prediction(f'results/arpabet_simple_{LLM}_good.csv')
    # per1_lora = evaluate_arpabet_prediction(f'results/arpabet_simple_lora-{LLM}_bad.csv')
    # per2_lora = evaluate_arpabet_prediction(f'results/arpabet_simple_lora-{LLM}_good.csv')
    acc_rhyme = evaluate_rhyme_prediction(f'results/rhyming_simple_{LLM}_word.csv')
    acc_rhyme_slash = evaluate_rhyme_prediction(f'results/rhyming_simple_{LLM}_slash.csv')
    # acc_rhyme_lora = evaluate_rhyme_prediction(f'results/rhyming_simple_lora-{LLM}.csv')
    print(f"Bad: {acc_syl1}, {per1}")
    print(f"Good: {acc_syl2}, {per2}")
    # print(f"Bad lora: {acc_syl1_lora}, {per1_lora}")
    # print(f"Good lora: {acc_syl2_lora}, {per2_lora}")
    print("rhyme: ", acc_rhyme)
    print("rhyme slash: ", acc_rhyme_slash)
    # print("rhyme lora: ", acc_rhyme_lora)
    # acc1, recall1, precision1 = evaluate_rhyme_prediction(f'results/rhyming_simple_{LLM}_bad.csv')
    # acc2, recall2, precision2 = evaluate_rhyme_prediction(f'results/rhyming_simple_{LLM}_good.csv')
    # print(f"Bad: {acc1}, {recall1}, {precision1}, {per1}, {acc_syl1}")
    # print(f"Good: {acc2}, {recall2}, {precision2}, {per2}, {acc_syl2}")
