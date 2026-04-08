import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
import random
import copy


IPA_TOKENS = ['<IPA>', '</IPA>']


# a possible list of prompts for rhyming awareness
RHYMING_PROMPTS = [
    "Does the word '{word1}' rhyme with the word '{word2}'?",
    "Is the word '{word1}' a rhyme for the word '{word2}'?",
    "Are the words '{word1}' and '{word2}' rhyming words?",
    "Are words '{word1}' and '{word2}' in the same rhyme scheme?",
    "Is the word '{word1}' in rhyme with the word '{word2}'?",
    "Does the word {word1} rhyme with the word {word2}?",
    "Is the word {word1} a rhyme for the word {word2}?",
    "Are the words {word1} and {word2} rhyming words?",
    "Are words {word1} and {word2} in the same rhyme scheme?",
    "Is the word {word1} in rhyme with the word {word2}?",
]


