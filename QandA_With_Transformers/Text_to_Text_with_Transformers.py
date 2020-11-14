"""
Purvang Lapsiwala
Description:
    This file contains the code for problems of Text to Text transfer and Question Answering using
    Transformer Encoder Block. (Transformers)
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
# Transformer

# Transformer Encoder


def FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes, mode, activation):
    """Returns a list of layers implementing a feed-forward block.
    Args:
        d_model: int:  depth of embedding
        d_ff: int: depth of feed-forward layer
        dropout: float: dropout rate (how much to drop out)
        dropout_shared_axes: list of integers, axes to share dropout mask
        mode: str: 'train' or 'eval'
        activation: the non-linearity in feed-forward layer
    Returns:
        A list of layers which maps vectors to vectors.
    """

    dropout_middle = tl.Dropout(rate=dropout,
                                shared_axes=dropout_shared_axes,
                                mode=mode)

    dropout_final = tl.Dropout(rate=dropout,
                               shared_axes=dropout_shared_axes,
                               mode=mode)

    ff_block = [
        # trax Layer normalization
        tl.LayerNorm(),
        # trax Dense layer using `d_ff`
        tl.Dense(d_ff),
        # activation() layer - you need to call (use parentheses) this func!
        activation(),
        # dropout middle layer
        dropout_middle,
        # trax Dense layer using `d_model`
        tl.Dense(d_model),
        # dropout final layer
        dropout_final,
    ]

    return ff_block


# Complete encoder block. we will use Feed Forward block just created above.

def EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                 mode, ff_activation, FeedForwardBlock=FeedForwardBlock):
    """
    Returns a list of layers that implements a Transformer encoder block.
    The input to the layer is a pair, (activations, mask), where the mask was
    created from the original source tokens to prevent attending to the padding
    part of the input.

    Args:
        d_model (int): depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        dropout_shared_axes (int): axes on which to share dropout mask.
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.
        FeedForwardBlock (function): A function that returns the feed forward block.
    Returns:
        list: A list of layers that maps (activations, mask) to (activations, mask).

    """

    # Attention block
    attention = tl.Attention(
        # dimension of the model
        d_feature=d_model,
        # number of attention heads
        n_heads=n_heads,
        # `dropout`
        dropout=dropout,
        # `mode`
        mode=mode
    )

    # calling function `FeedForwardBlock
    feed_forward = FeedForwardBlock(
        d_model,
        d_ff,
        dropout,
        dropout_shared_axes,
        mode,
        ff_activation
    )

    # Dropout block
    dropout_ = tl.Dropout(
        rate=dropout,
        shared_axes=dropout_shared_axes,
        mode=mode
    )

    encoder_block = [
        # `Residual` layer
        tl.Residual(
            tl.LayerNorm(),
            attention,
            dropout_,
        ),
        tl.Residual(
            feed_forward,
        ),
    ]
    return encoder_block


def TransformerEncoder(vocab_size=vocab_size,
                       n_classes=10,
                       d_model=512,
                       d_ff=2048,
                       n_layers=6,
                       n_heads=8,
                       dropout=0.1,
                       dropout_shared_axes=None,
                       max_len=2048,
                       mode='train',
                       ff_activation=tl.Relu,
                       EncoderBlock=EncoderBlock):
    """
    Returns a Transformer encoder model.
    The input to the model is a tensor of tokens.

    Args:
        vocab_size (int): vocab size. Defaults to vocab_size.
        n_classes (int): how many classes on output. Defaults to 10.
        d_model (int): depth of embedding. Defaults to 512.
        d_ff (int): depth of feed-forward layer. Defaults to 2048.
        n_layers (int): number of encoder/decoder layers. Defaults to 6.
        n_heads (int): number of attention heads. Defaults to 8.
        dropout (float): dropout rate (how much to drop out). Defaults to 0.1.
        dropout_shared_axes (int): axes on which to share dropout mask. Defaults to None.
        max_len (int): maximum symbol length for positional encoding. Defaults to 2048.
        mode (str): 'train' or 'eval'. Defaults to 'train'.
        ff_activation (function): the non-linearity in feed-forward layer. Defaults to tl.Relu.
        EncoderBlock (function): Returns the encoder block. Defaults to EncoderBlock.

    Returns:
        trax.layers.combinators.Serial: A Transformer model as a layer that maps
        from a tensor of tokens to activations over a set of output classes.
    """

    positional_encoder = [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode),
        tl.PositionalEncoding(max_len=max_len)
    ]

    # repeatation of Encoder block upto number of layers
    encoder_blocks = [EncoderBlock(d_model, d_ff, n_heads, dropout,
                                   dropout_shared_axes, mode, ff_activation) for _ in range(n_layers)]

    # Encoder Model
    return tl.Serial(
        tl.Branch(
            positional_encoder,
            tl.PaddingMask(),
        ),
        encoder_blocks,
        tl.Select([0], n_in=2),
        tl.LayerNorm(),
        tl.Mean(axis=1),
        tl.Dense(n_classes),
        tl.LogSoftmax(),
    )
