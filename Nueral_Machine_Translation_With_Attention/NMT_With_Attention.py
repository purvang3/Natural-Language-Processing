"""
Author: Purvang Lapsiwala
Description:
    This file contains the code to for Nueral Machine Translation model to convert Text in English language to
    German language using encoder , decoder and attention method
Packages: trax, numpy
"""

from termcolor import colored
import random
import numpy as np
import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training

# ====================================================================================================================

VOCAB_FILE = 'ende_32k.subword'
VOCAB_DIR = '/content/'
EOS = 1

# ====================================================================================================================

# Data Preparation

# TFDS api returns generator function yielding tuple of (inputs, target)
# keys return data in pair of specified languages.

train_data_fn = trax.data.TFDS('opus/medical',
                               data_dir='./data/',
                               keys=('en', 'de'),
                               eval_holdout_size=0.01,
                               train=True)

# Get generator function for the eval set
eval_data_fn = trax.data.TFDS('opus/medical',
                              data_dir='./data/',
                              keys=('en', 'de'),
                              eval_holdout_size=0.01,
                              train=False)

train_data = train_data_fn()
eval_data = eval_data_fn()

# using Tokenize method to convert text data into numpy array of integers
tokenized_train_data = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(train_data)
tokenized_eval_data = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(eval_data)


# generator helper function to append EOS to each sentence
def append_eos(data):
    for (inputs, targets) in data:
        inputs_with_eos = list(inputs) + [EOS]
        targets_with_eos = list(targets) + [EOS]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)


# append EOS to the train data
tokenized_train_data = append_eos(tokenized_train_data)

# append EOS to the eval data
tokenized_eval_data = append_eos(tokenized_eval_data)

# Filtering data to limit sentence length (use only if memory is constrain)
filtered_train_data = trax.data.FilterByLength(max_length=256, length_keys=[0, 1])(tokenized_train_data)
filtered_eval_data = trax.data.FilterByLength(max_length=512, length_keys=[0, 1])(tokenized_eval_data)


def tokenize(input_str, vocab_file=None, vocab_dir=None):
    """
    function to encode string data to int array
    Args:
        input_str: human readable text string
        vocab_file: vocab file which will be used as reference to tokenize text
        vocab_dir: path of vocab file

    Returns: [1, len(input_str)]

    """
    global EOS
    inputs = next(trax.data.tokenize(iter([input_str]), vocab_file=vocab_file, vocab_dir=vocab_dir))
    inputs = list(inputs) + [EOS]
    batch_inputs = np.reshape(np.array(inputs), [1, -1])
    return batch_inputs


def detokenize(integers, vocab_file=None, vocab_dir=None):
    """
    function to decode array of int to sentence
    Args:
        integers: array of integers
        vocab_file: vocab file which will be used as reference to detokenize text
        vocab_dir: path of vocab file

    Returns: input_string

    """
    global EOS
    integers = list(np.squeeze(integers))

    if EOS in integers:
        integers = integers[:integers.index(EOS)]

    return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir=vocab_dir)


# Bucketing operation batching tokenized sentences based on the sentence length
boundaries = [8, 16, 32, 64, 128, 256, 512]
batch_sizes = [256, 128, 64, 32, 16, 8, 4, 2]

train_batch_data = trax.data.BucketByLength(boundaries, batch_sizes, length_keys=[0, 1])(filtered_train_data)
eval_batch_data = trax.data.BucketByLength(boundaries, batch_sizes, length_keys=[0, 1])(filtered_eval_data)

# Add masking for the padding (0s) to make it bucket length length
train_batch_data = trax.data.AddLossWeights(id_to_mask=0)(train_batch_data)
eval_batch_data = trax.data.AddLossWeights(id_to_mask=0)(eval_batch_data)
