import pandas as pd
import numpy as np
import torch
import os
import re
import csv
import sys
import itertools
from tokenizer import split_into_sentences      # from Miðeind
from reynir import Greynir
from lexicalrichness import LexicalRichness

# List the names of files in the three directories
# Childhood Home - Picture Description - Travel Planning
CH_files = sorted(os.listdir("CUT-transcript-parts/CH-transcripts/"))
PD_files = sorted(os.listdir("CUT-transcript-parts/PD-transcripts/"))
TP_files = sorted(os.listdir("CUT-transcript-parts/TP-transcripts/"))

# Clean up the transcript files (except for corrections (#))
def hreinsun(files, mappa):
    for file in files:
        with open(f'CUT-transcript-parts/{mappa}/{file}', 'r') as skra:
            f = open(f'Acode_clean/{mappa}/{file}', 'w+')
            for line in skra:
                # Only lines that begin with V (for viðmælandi)
                if line[0] == "V":
                    # Remove unnecessary symbols, words (hikorð/innskotsorð) and unfinished words (-)
                    res = re.sub(r'\{(.*?)\}|\+|\(|\)|\~|\*(hérna|þarna)\*|\b(öö|ömm|mm|uu|mhm|hmm|umm|ee|aa|ja)\b|\s?\w+-', '', line, flags=re.IGNORECASE)
                    # Remove repetitions (í= í= í -> í)
                    res = re.sub(r'\b(\w+(?:\s+\w+)*)\b(?:\s*=\s*\1)+', r'\1', res, flags=re.IGNORECASE)
                    # Tokenization
                    for setning in split_into_sentences(res[3:]):
                        res2 = re.sub(r'\.|\?|,', '', setning)
                        f.write(res2)
                        f.write('\n')
            f.close()

hreinsun(CH_files, "CH-transcripts")
hreinsun(PD_files, "PD-transcripts")
hreinsun(TP_files, "TP-transcripts")

# Prepare the model for PoS tagging (https://github.com/mideind/GreynirSeq/blob/main/src/greynirseq/nicenlp/examples/pos/README.md)
model = torch.hub.load("mideind/GreynirSeq:main", "icebert-pos")
model.eval()

# PoS tagging
def pos(files, mappa):
    merki = []
    for file in files:
        with open(f'Acode_clean/{mappa}/{file}', 'r') as skra:
            f = open(f'Acode_tagged/{mappa}/{file}', 'w+')
            for line in skra:
                labels = model.predict_ifd_labels([line])
                for word in labels[0]:
                    if len(merki) < len(line.split()):
                        merki.append(word)
                        f.write(word + " ")
                f.write('\n')
                merki.clear()
            f.close()

pos(CH_files, "CH-transcripts")
pos(PD_files, "PD-transcripts")
pos(TP_files, "TP-transcripts")


# Lists for results
participants = []
samples = []
word_count = []
noun_count = []             # Nafnorð
pronoun_count = []          # Fornöfn
preposition_count = []      # Forsetningar
adverb_count = []           # Atviksorð
verb_count = []             # Sagnorð
conjunction_count = []      # Samtengingar
persónubeygðar = []         # Persónubeygðar sagnir
participle_count = []       # Lýsingarháttur
subjunctive_count = []      # Viðtengingarháttur
unfinished_words = []       # Ókláruð orð
corrections = []            # Leiðréttingar
fallorð = []                # Words inflected for case
fallorð_nf = []             # Nominative
fallorð_þf = []             # Accusative
fallorð_þgf = []            # Dative
fallorð_ef = []             # Genitive
typetokenratio = []         # Type/token ratio (moving average)
average_sentence_length = []      # Meðallengd setninga
word_freqs = []             # Tíðni orða


# Count unfinished words and corrections along with moving average TTR
# MATTR is calculated with https://pypi.org/project/lexicalrichness/ which is based on Covington and McFall 2010 (https://www.tandfonline.com/doi/full/10.1080/09296171003643098?casa_token=Fagv-eG6WoAAAAAA%3AaIEPDf_loZkUeAIj_dUw17SONc_q45Ud5wJ6O2yLTrPc4mC_yW0KWxz62C0Ilh9_DY5pl1iAIuo94go)
def talning(files, mappa):
    for file in files:
        ókláruð = 0
        lagfæringar = 0
        with open(f'CUT-transcript-parts/{mappa}/{file}', 'r') as skra:
            texti = skra.read()
            lex = LexicalRichness(texti)
            mattr = lex.mattr(100)
            skra.seek(0)
            for line in skra:
                if line[0] == "V":
                    for stafur in line:
                        if stafur == "-":
                            ókláruð += 1
                        if stafur == "#":
                            lagfæringar += 1
        unfinished_words.append(ókláruð)
        corrections.append(lagfæringar)
        typetokenratio.append(mattr)

talning(CH_files, "CH-transcripts")
talning(PD_files, "PD-transcripts")
talning(TP_files, "TP-transcripts")

# Count nouns, pronouns, verbs, etc.
def urvinnsla(files, mappa):
    for file in files:
        participants.append(file[3:-10])
        samples.append(file[:2])
        words = 0
        nouns = 0
        pronouns = 0
        prepositions = 0
        adverbs = 0
        verbs = 0
        conjunctions = 0
        pers_beygðar = 0
        participle = 0
        subjunctive = 0
        fo = 0
        fo_nf = 0
        fo_þf = 0
        fo_þgf = 0
        fo_ef = 0
        counts = []
        with open(f'Acode_tagged/{mappa}/{file}', 'r') as skra:
            for line in skra:
                counts.append(len(line.split()))
                for word in (line.split()):
                    words += 1
                    if word[0] == "n":
                        nouns += 1
                        fo += 1
                        if word[3] == "n":
                            fo_nf += 1
                        if word[3] == "o":
                            fo_þf += 1
                        if word[3] == "þ":
                            fo_þgf += 1
                        if word[3] == "e":
                            fo_ef += 1
                    if word[0] == "f":
                        pronouns += 1
                        fo += 1
                        if word[4] == "n":
                            fo_nf += 1
                        if word[4] == "o":
                            fo_þf += 1
                        if word[4] == "þ":
                            fo_þgf += 1
                        if word[4] == "e":
                            fo_ef += 1
                    if word[:2] == "af":
                        prepositions += 1
                    if word[:2] == "aa":
                        adverbs += 1
                    if word[0] == "s":
                        verbs += 1
                        if word[1] == "þ" or word[1] == "l":
                            participle += 1
                        if word[1] == "v":
                            subjunctive += 1
                        tolur = set('123')
                        if any((tala in tolur) for tala in word):
                            pers_beygðar += 1
                    if word[0] == "c":
                        conjunctions += 1
                    if word[0] == "l" or word[0] == "g":
                        fo += 1
                        if word[3] == "n":
                            fo_nf += 1
                        if word[3] == "o":
                            fo_þf += 1
                        if word[3] == "þ":
                            fo_þgf += 1
                        if word[3] == "e":
                            fo_ef += 1
                    if word[0] == "t" and len(word) > 2:
                        fo += 1
                        if word[4] == "n":
                            fo_nf += 1
                        if word[4] == "o":
                            fo_þf += 1
                        if word[4] == "þ":
                            fo_þgf += 1
                        if word[4] == "e":
                            fo_ef += 1
        average_sentence_length.append(float(sum(counts)/len(counts)))
        word_count.append(words)
        noun_count.append(nouns)
        pronoun_count.append(pronouns)
        preposition_count.append(prepositions)
        adverb_count.append(adverbs)
        verb_count.append(verbs)
        conjunction_count.append(conjunctions)
        persónubeygðar.append(pers_beygðar)
        participle_count.append(participle)
        subjunctive_count.append(subjunctive)
        fallorð.append(fo)
        fallorð_nf.append(fo_nf)
        fallorð_þf.append(fo_þf)
        fallorð_þgf.append(fo_þgf)
        fallorð_ef.append(fo_ef)

urvinnsla(CH_files, "CH-transcripts")
urvinnsla(PD_files, "PD-transcripts")
urvinnsla(TP_files, "TP-transcripts")

# Mæling á tíðni orða / Word frequency
# Read the frequency list for Risamálheildin (Icelandic Gigaword Corpus) to a dict (https://repository.clarin.is/repository/xmlui/handle/20.500.12537/314)
# The lemma of a word and its word class (in a tuple) are the key, its frequency is the value
def read_freq_list(freq_file):
    word_frequency = {}
    with open(freq_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            word_pos = (row[0], row[1])
            frequency = int(row[2])
            word_frequency[word_pos] = frequency
    return word_frequency
word_freq_dict = read_freq_list("giga_simple_freq.tsv")

# Got the lemmas with Nefnir (https://github.com/jonfd/nefnir)
def word_freq(files, mappa1, mappa2, word_freq_dict):
    for file in files:
        file_word_fr = []
        # Read the lemmas and PoS tags into lists
        lemmas_list = [line.split()[2] for line in open(mappa1+file, 'r')]
        pos_list = [tag[:2] if tag.startswith('n') else tag[0] for line in open(mappa2+file, 'r') for tag in line.split()]
        # Look up the (lemma, word class) in the frequency dict
        for item in itertools.zip_longest(lemmas_list, pos_list):
            if item in word_freq_dict:
                file_word_fr.append(word_freq_dict[item])
            # Add 0 if a word is not in the frequency dict
            else:
                file_word_fr.append(0)
        #word_freqs.append(sum(file_word_freqs)/len(file_word_freqs))    # Mean / meðaltal
        word_freqs.append(np.median(file_word_fr))                       # Median / miðgildi
        
word_freq(CH_files, "Acode_lemmun/CH-lemmað/", "Acode_tagged/CH-transcripts/", word_freq_dict)
word_freq(PD_files, "Acode_lemmun/PD-lemmað/", "Acode_tagged/PD-transcripts/", word_freq_dict)
word_freq(TP_files, "Acode_lemmun/TP-lemmað/", "Acode_tagged/TP-transcripts/", word_freq_dict)

# Export the results (pandas DataFrame) to Excel
# One sheet for cut transcript parts (Childhood Home - Picture Description - Travel Planning) and another for full transcripts
columns1 = ['Participant', 'Sample', 'Word count', 'Noun count', 'Pronoun count', 'Preposition count', 'Adverb count', 'Verb count', 'Person infl. verbs', 'Participle count', 'Subjunctive count', 'Conjunction count', 'Unfinished words', 'Corrections', 'Fallorð', 'Fallorð í nf.', 'Fallorð í þf.', 'Fallorð í þgf.', 'Fallorð í ef.', 'TTR (ma)', 'Avg. sentence length', 'Word frequency']
df1 = pd.DataFrame(list(zip(participants, samples, word_count, noun_count, pronoun_count, preposition_count, adverb_count, verb_count, persónubeygðar, participle_count, subjunctive_count, conjunction_count, unfinished_words, corrections, fallorð, fallorð_nf, fallorð_þf, fallorð_þgf, fallorð_ef, typetokenratio, average_sentence_length, word_freqs)), columns=columns1)
df1 = df1.sort_values(by=['Participant', 'Sample'])

columns2 = ['Participant', 'Word count', 'Noun count', 'Pronoun count', 'Preposition count', 'Adverb count', 'Verb count', 'Person infl. verbs', 'Participle count', 'Subjunctive count', 'Conjunction count', 'Unfinished words', 'Corrections', 'Fallorð', 'Fallorð í nf.', 'Fallorð í þf.', 'Fallorð í þgf.', 'Fallorð í ef.', 'TTR (ma)', 'Avg. sentence length', 'Word frequency']
df2 = df1.groupby('Participant').sum().reset_index()

df2 = df1.groupby('Participant').agg({'Word count': 'sum',
                               'Noun count': 'sum',
                               'Pronoun count': 'sum',
                               'Preposition count': 'sum',
                               'Adverb count': 'sum',
                               'Verb count': 'sum',
                               'Persónubeygðar sagnir': 'sum',
                               'Participle count': 'sum',
                               'Subjunctive count': 'sum',
                               'Conjunction count': 'sum',
                               'Unfinished words': 'sum',
                               'Corrections': 'sum',
                               'Fallorð': 'sum',
                               'Fallorð í nf.': 'sum',
                               'Fallorð í þf.': 'sum',
                               'Fallorð í þgf.': 'sum',
                               'Fallorð í ef.': 'sum',
                               'TTR (moving average)': 'mean',
                               'Average sentence length': 'mean',
                               'Word frequency': 'mean'
                               }).reset_index()


with pd.ExcelWriter('Acode-POS.xlsx') as writer:
    df2.to_excel(writer, sheet_name='FULL', index=False)
    df1.to_excel(writer, sheet_name='CUT', index=False)

