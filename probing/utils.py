import torch
import panphon
import panphon.distance
import pandas as pd

VOWEL = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
    'EH', 'ER', 'EY', 'IH', 'IY', 'OW',
    'OY', 'UH', 'UW',
]

CONSONANT = [
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH',
    'JH', 'K', 'L', 'M', 'N', 'NG', 'P',
    'R', 'S', 'SH', 'T', 'TH', 'V', 'W',
    'Y', 'Z', 'ZH'
]

arpabet_to_ipa = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɜː',
    'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
    'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
    'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w',
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ', ' ': ' '
}

def first_vowel(arpa):
    for p in arpa:
        if p in VOWEL:
            return p
    return None

def label_to_ARPR(labels):
    """
    convert a list of integer to ARPR
    labels : List[int]
    """
    # Combine phonemes and create mapping
    phoneme_list = VOWEL + CONSONANT  # Total of 39 phonemes
    int_to_phoneme = {idx + 1 : phoneme for idx, phoneme in enumerate(phoneme_list)}
    int_to_phoneme[0] = " "
    ARPRbet = []
    for i in labels:
        ARPRbet.append(int_to_phoneme[i])
    return ARPRbet

def ARPR_to_label(ARPRs):
    """
    convert a list of ARPRbets to labels
    ARPRs : List[ARPRbet]
    """
    # Combine phonemes and create mapping
    phoneme_list = VOWEL + CONSONANT  # Total of 39 phonemes
    phoneme_to_int = {phoneme : idx + 1 for idx, phoneme in enumerate(phoneme_list)}
    labels = []
    for ARPRbet in ARPRs:
        labels.append(phoneme_to_int[ARPRbet])
    return labels

def feature_edit_distance(p1, p2, dst):
    """
    calculate the feature edit distance between two list of ARPRbet
    p1: List[ARPRbet]
    p2: List[ARPRbet]
    dst: Distance
    """
    # dst = panphon.distance.Distance()
    ipa1, ipa2 = [], []
    for i in range(len(p1)):
        ipa1.append(arpabet_to_ipa[p1[i]])
    for j in range(len(p2)):
        ipa2.append(arpabet_to_ipa[p2[j]])
    p1_string = ''.join(ipa1)
    p2_string = ''.join(ipa2)
    # return dst.hamming_feature_edit_distance(p1_string, p2_string)
    return dst.feature_edit_distance_div_maxlen(p1_string, p2_string)


def drop_trailing_zeros_tensor(tensor):
    non_zero_indices = torch.nonzero(tensor, as_tuple=True)[0]
    if len(non_zero_indices) > 0:
        last_non_zero = non_zero_indices[-1].item()
        return tensor[:last_non_zero + 1]
    else:
        return torch.tensor([], dtype=tensor.dtype, device=tensor.device)
    
def drop_all_zeros_tensor(tensor):
    # Find indices of all non-zero elements
    non_zero_indices = torch.nonzero(tensor, as_tuple=True)[0]
    if len(non_zero_indices) > 0:
        return tensor[non_zero_indices]
    else:
        return torch.tensor([], dtype=tensor.dtype, device=tensor.device)
    

# Define the pronunciation to vector function
def pronunciation_to_vector(pronunciation, max_length=8):
    # Split the pronunciation into phonemes
    phoneme_list = VOWEL + CONSONANT  # Total of 39 phonemes
    phoneme_to_int = {phoneme : idx + 1 for idx, phoneme in enumerate(phoneme_list)}
    # Map each phoneme to its corresponding integer
    phoneme_ints = [phoneme_to_int.get(ph, 0) for ph in pronunciation]  # Use 0 for unknown phonemes
    # Pad the vector with zeros if necessary
    if len(phoneme_ints) < max_length:
        phoneme_ints.extend([0] * (max_length - len(phoneme_ints)))
    else:
        phoneme_ints = phoneme_ints[:max_length]  # Truncate if longer than max_length
    return phoneme_ints
