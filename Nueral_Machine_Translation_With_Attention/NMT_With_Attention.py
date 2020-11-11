"""
Author: Purvang Lapsiwala
Description:
    This file contains the code for Nueral Machine Translation model to convert Text in English language to
    German language using encoder , decoder and attention method. (Encoder Decoder attention)
Packages: trax, numpy
"""

from termcolor import colored
import random
import os
import numpy as np
import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers import PureAttention
from trax.supervised import training

# ====================================================================================================================

VOCAB_FILE = 'ende_32k.subword'
VOCAB_DIR = "./Nueral_Machine_Translation_With_Attention/data/"
EOS = 1

# ====================================================================================================================

# Data Preparation

# TFDS api returns generator function yielding tuple of (inputs, target)
# keys return data in pair of specified languages.

train_data_fn = trax.data.TFDS('opus/medical',
                               data_dir='./Nueral_Machine_Translation_With_Attention/data/data/',
                               keys=('en', 'de'),
                               eval_holdout_size=0.01,
                               train=True)

# Get generator function for the eval set
eval_data_fn = trax.data.TFDS('opus/medical',
                              data_dir='./Nueral_Machine_Translation_With_Attention/data/data/',
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

# Bucketing operation batching tokenized sentences based on the sentence length
boundaries = [8, 16, 32, 64, 128, 256, 512]
batch_sizes = [256, 128, 64, 32, 16, 8, 4, 2]

train_batch_data = trax.data.BucketByLength(boundaries, batch_sizes, length_keys=[0, 1])(filtered_train_data)
eval_batch_data = trax.data.BucketByLength(boundaries, batch_sizes, length_keys=[0, 1])(filtered_eval_data)

# Add masking for the padding (0s) to make it bucket length length
train_batch_data = trax.data.AddLossWeights(id_to_mask=0)(train_batch_data)
eval_batch_data = trax.data.AddLossWeights(id_to_mask=0)(eval_batch_data)


# ====================================================================================================================
# Helper function to tokenize and detokenize individual sentences

def tokenize(input_str, vocab_file=None, vocab_dir=None, EOS_=1):
    """
    function to encode string data to int array
    Args:
        input_str: human readable text string
        vocab_file: vocab file which will be used as reference to tokenize text
        vocab_dir: path of vocab file

    Returns: [1, len(input_str)]
    """
    # tokenize method operated with stream. so iter and next is used to get stream from array
    inputs = next(trax.data.tokenize(iter([input_str]), vocab_file=vocab_file, vocab_dir=vocab_dir))
    inputs = list(inputs) + [EOS_]
    batch_inputs = np.reshape(np.array(inputs), [1, -1])
    return batch_inputs


def detokenize(integers, vocab_file=None, vocab_dir=None, EOS_=1):
    """
    function to decode array of int to sentence
    Args:
        integers: array of integers
        vocab_file: vocab file which will be used as reference to detokenize text
        vocab_dir: path of vocab file

    Returns: input_string
    """
    integers = list(np.squeeze(integers))

    if EOS_ in integers:
        integers = integers[:integers.index(EOS_)]

    return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir=vocab_dir)

# ====================================================================================================================

# Attention Layer

# input_encoder_function


def input_encoder_fn(input_vocab_size, d_model, n_encoder_layers):
    """
    convert tokenize sentence into encoder activations gives keys and value for attention
    Args:
        input_vocab_size: int: vocab size of the input
        d_model: int:  dimention of embedding (n_units in the LSTM cell)
        n_encoder_layers: int: number of LSTM layers in the encoder
    Returns:
        tl.Serial: The input encoder
    """
    input_encoder = tl.Serial(
        # create an embedding layer to convert tokens to vectors
        tl.Embedding(vocab_size=input_vocab_size, d_feature=d_model), #(B,L) -> (B, D)

        # feed the embeddings to the LSTM layers. It is a stack of n_encoder_layers LSTM layers
        [tl.LSTM(n_units=d_model) for _ in range(n_encoder_layers)]
    )
    return input_encoder

# Pre-attention Decoder
# it will return query which will be used for attention layer


def pre_attention_decoder_fn(mode, target_vocab_size, d_model):
    """
    Pre-attention decoder runs on the targets and creates
    activations that are used as queries in attention.
    Args:
        mode: str: 'train' or 'eval'
        target_vocab_size: int: vocab size of the target
        d_model: int:  depth of embedding (n_units in the LSTM cell)
    Returns:
        tl.Serial: The pre-attention decoder
    """
    pre_attention_decoder = tl.Serial(

        # shift right to insert start-of-sentence token and implement
        # teacher forcing during training
        tl.ShiftRight(mode=mode),

        # run an embedding layer to convert tokens to vectors
        tl.Embedding(vocab_size=target_vocab_size, d_feature=d_model),

        # feed to an LSTM layer
        tl.LSTM(n_units=d_model)
    )
    return pre_attention_decoder


# Preparation of attention Input


def prepare_attention_input(encoder_activations, decoder_activations, inputs):
    """
    function will prepare K, Q, V and M for attention layer.
    Args:
        encoder_activations fastnp.array(batch_size, padded_input_length, d_model): output from the input encoder
        decoder_activations fastnp.array(batch_size, padded_input_length, d_model): output from the pre-attention decoder
        inputs fastnp.array(batch_size, padded_input_length): padded input tokens
    Returns:
        queries, keys, values and mask for attention.
    """
    # set the keys and values to the encoder activations
    keys = encoder_activations
    values = encoder_activations

    # set the queries to the decoder activations
    queries = decoder_activations

    # generate the mask to distinguish real tokens from padding
    mask = inputs != 0

    # add axes to the mask for attention heads and decoder length.
    mask = fastnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1])) # (B, heads, decoder_len, encoder_len)
    # broadcast so mask shape is [batch size, attention heads, decoder-len, encoder-len].
    # note: for this assignment, attention heads is set to 1.
    mask = mask + fastnp.zeros((1, 1, decoder_activations.shape[1], 1))

    return queries, keys, values, mask


def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode='train'):
    """Returns a layer that maps (q, k, v, mask) to (activations, mask).
    Args:
      d_feature: Depth/dimensionality of feature embedding.
      n_heads: Number of attention heads.
      dropout: Probababilistic rate for internal dropout applied to attention
          activations (based on query-key pairs) before dotting them with values.
      mode: Either 'train' or 'eval'.
    """
    return cb.Serial(
        cb.Parallel(
            core.Dense(d_feature),
            core.Dense(d_feature),
            core.Dense(d_feature),
        ),
        PureAttention(n_heads=n_heads, dropout=dropout, mode=mode),
        core.Dense(d_feature),
    )
