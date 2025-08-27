system_message = """

You are a phonology expert, and you can do Grapheme to Phoneme very well,
You are only allowed to use the following phonemes:

Vowels
- AA: as in *father* (/ɑ/)
- AE: as in *cat* (/æ/)
- AH: as in *cup* (/ʌ/)
- AO: as in *caught* (/ɔ/)
- AW: as in *now* (/aʊ/)
- AY: as in *my* (/aɪ/)
- EH: as in *bed* (/ɛ/)
- ER: as in *bird* (/ɝ/ or /ɚ/)
- EY: as in *say* (/eɪ/)
- IH: as in *sit* (/ɪ/)
- IY: as in *see* (/i/)
- OW: as in *go* (/oʊ/)
- OY: as in *boy* (/ɔɪ/)
- UH: as in *book* (/ʊ/)
- UW: as in *too* (/u/)

Consonants
- B: as in *boy* (/b/)
- CH: as in *chop* (/tʃ/)
- D: as in *dog* (/d/)
- DH: as in *this* (/ð/)
- F: as in *fish* (/f/)
- G: as in *go* (/ɡ/)
- HH: as in *hat* (/h/)
- JH: as in *judge* (/dʒ/)
- K: as in *key* (/k/)
- L: as in *look* (/l/)
- M: as in *man* (/m/)
- N: as in *no* (/n/)
- NG: as in *sing* (/ŋ/)
- P: as in *pat* (/p/)
- R: as in *red* (/r/)
- S: as in *see* (/s/)
- SH: as in *shoe* (/ʃ/)
- T: as in *top* (/t/)
- TH: as in *thin* (/θ/)
- V: as in *van* (/v/)
- W: as in *we* (/w/)
- Y: as in *yes* (/j/)
- Z: as in *zoo* (/z/)
- ZH: as in *measure* (/ʒ/)

System Note: When transcribing words, provide each phoneme as a space-separated sequence, ensuring precise pronunciation.
And Only use the ARPAbet symbols listed above.
"""
# And first consider the IPA of the word, and then provide the ARPAbet transcription by transforming the IPA symbols into their ARPAbet equivalents.


prompt_g2p_simple = (
    "ARPAbet is a phonetic transcription system used to represent the pronunciation of words. Below are the ARPAbet symbols:\n\n"

    "Vowels:\n"
    "AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW\n\n"

    "Consonants:\n"
    "B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, P, R, S, SH, T, TH, V, W, Y, Z, ZH\n\n"

    "Provide ARPAbet transcriptions using only the symbols above, add space between each phoneme.\n"
    "For example:\n"
    "Word: decide\n"
    "ARPAbet: K AE T\n"
    "Word: dog\n"
    "ARPAbet: D AO G\n\n"
    "Now, transcribe the following word, output the answer as 'ARPAbet: <phoneme sequence>' and stop generating after the answer.\n"
    "Word: {word}\n"
)



prompt_g2p_cot = (
    "To determine the ARPAbet transcription for a word, follow these steps:\n"
    "1. Identify the individual phonemes (speech sounds) in the word.\n"
    "2. Map each phoneme to its corresponding ARPAbet symbol.\n"
    "3. Combine the symbols to form the ARPAbet transcription.\n"
    "\n"
    "Let's find the ARPAbet transcription for the word {word}.\n"
    "Step-by-step reasoning:"
)

prompt_g2p_icl = (
    "Here are some examples of ARPAbet transcriptions:\n"
    "\n"
    "Example 1:\n"
    "Word: cat\n"
    "ARPAbet: K AE T\n"
    "\n"
    "Example 2:\n"
    "Word: dog\n"
    "ARPAbet: D AO G\n"
    "\n"
    "Now, give the ARPAbet transcription for the word {word}:"
)


### Syllables ###

system_message_syllables = """
You are a phonology expert, you can do syllabification very well.

Syllable is a unit of pronunciation having one vowel sound, with or without surrounding consonants,
forming the whole or a part of a word.
"""

prompt_syllables_word_simple = (
    "Count the number of syllables in the word: '{word}'\n."
    "Give the answer in the format:\n"
    "Answer: <number of syllables>\n"
)

prompt_syllables_word_one_shot = (
    "Count the number of syllables in the word: '{word}'\n"
    "For example, the word 'apple' has 2 syllables\n"
    "Output your answer as 'Answer: <number of syllables>'"
)

prompt_syllables_word_cot = (
    "To count the number of syllables in a word, follow these steps:\n"
    "1. Divide the word into its individual vowel sounds.\n"
    "2. Count each distinct vowel sound as one syllable.\n"
    "3. Adjust for any silent vowels or combined sounds that do not form separate syllables.\n"
    "\n"
    "Let's count the syllables in the word: '{word}'.\n"
    "Step-by-step reasoning:"
)


prompt_syllables_word_icl = (
    "Count the number of syllables in a word based on its vowel sounds. Here are some examples:\n"
    "\n"
    "Example 1:\n"
    "Word: 'apple'\n"
    "Syllables: 2\n"
    "\n"
    "Example 2:\n"
    "Word: 'banana'\n"
    "Syllables: 3\n"
    "\n"
    "Example 3:\n"
    "Word: 'tree'\n"
    "Syllables: 1\n"
    "\n"
    "Now, count the syllables in the word: '{word}'."
)

prompt_syllables_word_count = (
    "First count number of letters in the word, and then count the number of syllables in the word: {word}"
)

prompt_syllables_word_IPA = (
    "consider the IPA of the word first, and then count the number of syllables in the word: {word}"
)


### Rhyming ###

prompt_template_icl_rhyming = (
    "Rhyming words are words that have the same ending sound. Determine if the following two words rhyme. "
    "Answer strictly as 'Answer: True' if they rhyme and 'Answer: False' if they do not. Here are some examples:\n\n"
    "Example 1:\n"
    "Words: cat, hat\n"
    "Answer: True\n\n"
    "Example 2:\n"
    "Words: sun, fun\n"
    "Answer: True\n\n"
    "Example 3:\n"
    "Words: boy, girl\n"
    "Answer: False\n\n"
    "Example 4:\n"
    "Words: red, blue\n"
    "Answer: False\n\n"
    "Now, analyze the following pair of words and provide your answer in the same format.\n"
    "Words: {word1}, {word2}\n"
    "Answer:"
)

prompt_template_cot_rhyming = (
    "Rhyming words are words that have the same ending sound. Let's determine if the two words rhyme by reasoning step by step.\n\n"
    "Word 1: {word1}\n"
    "Word 2: {word2}\n\n"
    "Step 1: Analyze the ending sounds of both words.\n"
    "Step 2: Compare if the ending sounds are identical\n"
    "Step 3: Based on the comparison, decide if the words rhyme.\n\n"
    "Do these words rhyme? Let's think step by step:"
)

prompt_template_simple_rhyming = (
    "Rhyming words are words that have the same ending sound. Determine if the following two words are in rhyme.\n{word1}, {word2}\n"
    "Give the answer as 'Answer: True' if they rhyme and 'Answer: False' if they do not."
)


prompt_template_simple2_rhyming = (
    "Rhyming words are words that have the same ending sound. Is word {word1} in rhyme with word {word2}?\n"
    "Give the answer as 'Answer: True' if they rhyme and 'Answer: False' if they do not."
)
