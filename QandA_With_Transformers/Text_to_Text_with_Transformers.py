"""
Author: Purvang Lapsiwala
Description:
    This file contains the code for Text to Text transfer and Question Answering using
    Bidirectional Encoder Decoder Model (BERT)
Packages: trax, numpy
"""

import ast
import string
import textwrap
import itertools
import numpy as np
import trax
from trax import layers as tl
from trax.supervised import decoding

PAD, EOS, UNK = 0, 1, 2
# Set random seed
np.random.seed(42)

wrapper = textwrap.TextWrapper(width=70)

# we will use c4 dataset from common crawl dataset.

example_jsons: [dict] = list(map(ast.literal_eval, open('QandA_With_Transformers/data/data.txt')))

# print(f'example: \n\n{example_jsons[0]} \n')

# ====================================================================================================================
# Only use to create input and targets from unlabelled data for pre training.

# We will use pre training approach to train model with unlabelled data.
# Steps to convert text in to inputs and labels

natural_language_texts: [string] = [example_json['text'] for example_json in example_jsons]


def detokenize(np_array):
    return trax.data.detokenize(
        np_array,
        vocab_type='sentencepiece',
        vocab_file='QandA_With_Transformers/data/sentencepiece.model',
        vocab_dir='.')


def tokenize(s):
    # tokenize method operated with stream. so iter and next is used to get stream from array
    return next(trax.data.tokenize(
        iter([s]),
        vocab_type='sentencepiece',
        vocab_file='QandA_With_Transformers/data/sentencepiece.model',
        vocab_dir='.'))


# Creating inputs and targets from Text. T5 uses id as sentinels at the end of vocab file
vocab_size = trax.data.vocab_size(
    vocab_type='sentencepiece',
    vocab_file='QandA_With_Transformers/data/sentencepiece.model',
    vocab_dir='.')


def get_sentinels(vocab_size=vocab_size, display=False):
    string.ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    sentinels = {}
    for i, char in enumerate(reversed(string.ascii_letters), 1):
        decoded_text = detokenize([vocab_size - i])
        sentinels[decoded_text] = f'<{char}>'
        if display:
            print(f'The sentinel is <{char}> and the decoded token is:', decoded_text)

    return sentinels


def pretty_decode(encoded_str_list):
    global vocab_size
    if isinstance(encoded_str_list, (str, bytes)):
        # for token, char in sentinels.items():
        for token, char in get_sentinels(vocab_size=vocab_size).items():
            encoded_str_list = encoded_str_list.replace(token, char)
        return encoded_str_list

    # for tokenize text, we first need to convert into human readable text and then replace sentinels in text as
    # required by T5 model
    return pretty_decode(detokenize(encoded_str_list))


######
# Tokenize and Masking

def tokenize_and_mask(text, vocab_size=vocab_size, noise=0.15, randomizer=np.random.uniform, tokenize=tokenize):
    """Tokenizes and masks a given input.

    Args:
        text: Text input.
        vocab_size: Size of the vocabulary. Defaults to vocab_size.
        noise: Probability of masking a token. Defaults to 0.15.
        randomizer: Function that generates random values. Defaults to np.random.uniform.
        tokenize: Tokenizer function.

    Returns:
        tuple: Tuple of lists of integers associated to inputs and targets.
    """

    cur_sentinel_num = 0
    inps = []
    targs = []

    # prev_no_mask is True if the previous token was NOT masked, False otherwise
    # set prev_no_mask to True
    prev_no_mask = True

    # loop through tokenized `text`
    for token in tokenize(text):
        # check if the `noise` is greater than a random value else add token to input
        if randomizer() < noise:
            # check to see if the previous token was not masked
            # and add sentinel id to inp and target

            if prev_no_mask == True:  # add new masked token at end_id
                # number of masked tokens increases by 1
                cur_sentinel_num += 1
                # compute `end_id` by subtracting current sentinel value out of the total vocabulary size
                end_id = vocab_size - cur_sentinel_num
                # append `end_id` at the end of the targets
                targs.append(end_id)
                # append `end_id` at the end of the inputs
                inps.append(end_id)
            # append `token` at the end of the targets
            targs.append(token)
            # set prev_no_mask accordingly
            prev_no_mask = False

        else:  # don't have two masked tokens in a row
            # append `token ` at the end of the inputs
            inps.append(token)
            # set prev_no_mask accordingly
            prev_no_mask = True

    return inps, targs


# creating dataset
inputs_targets_pairs = [tokenize_and_mask(text) for text in natural_language_texts]

# ====================================================================================================================
