from tokenize import Token
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import sys
import string

if sys.version_info[0] == 3 and sys.version_info[1] == 10:
    from keras.preprocessing.sequence import pad_sequences
else:
    from keras.utils import pad_sequences

import tensorflow_datasets as tfds


import pandas as pd
import numpy as np
import io
import os

def split_data(data, rate, x_col, y_col):
    size = int(len(data)*rate)

    x_train, y_train = data[x_col][:size], np.array(data[y_col][:size])
    x_val, y_val = data[x_col][size:], np.array(data[y_col][size:])

    return (x_train, y_train), (x_val, y_val)

    
def text_tokenizer(data, oov_toke="<OOV>",  vocab_size=None, max_subword_length = -1):
    if type(data) == np.ndarray :
        sentences = data.tolist()
    else:
        sentences = data

    if max_subword_length == -1:
        tokenizer = Tokenizer(oov_token=oov_toke, num_words=vocab_size)
        tokenizer.fit_on_texts(sentences)

    else:
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(sentences, vocab_size, max_subword_length=max_subword_length)

    return tokenizer

def preprocess(text, text_tokenizer, padding_type='post', trunc_type='post', max_length=None):
    if type(text) == np.ndarray :
        print(0)
        sentences = text.tolist()
    else:
        sentences = text

    tokenizer = text_tokenizer
    if str(type(tokenizer))=="<class 'keras.preprocessing.text.Tokenizer'>":
        sequences = tokenizer.texts_to_sequences(sentences)

    elif str(type(tokenizer))=="<class 'tensorflow_datasets.core.deprecated.text.subword_text_encoder.SubwordTextEncoder'>":
        sequences = []
        for i, sentence in enumerate(sentences):
            sequences.append(tokenizer.encode(sentence))

    padded = pad_sequences(sequences=sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)

    return padded

def reversed_word_index(word_index):
    return dict([(value, key) for (key, value) in word_index.items()])

def decode_text(text, reversed_word_index):
    return ' '.join([reversed_word_index.get(i, '?') for i in text])


def save_embeddings(directory, embedding_layer, model, vocab_size, tokenizer, reverse_word_index=None):
    # First get the weights of the embedding layer
    e = model.layers[embedding_layer]
    weights = e.get_weights()[0]

    # shape: (vocab_size, embedding_dim)
    print(weights.shape)

    vec_path = os.path.join(directory, 'vecs.tsv')
    meta_path = os.path.join(directory, 'meta.tsv')
    # Write out the embedding vectors and metadata
    out_v = io.open(vec_path, 'w', encoding='utf-8')
    out_m = io.open(meta_path, 'w', encoding='utf-8')

    if str(type(tokenizer))=="<class 'keras.preprocessing.text.Tokenizer'>":
        for word_num in range(1, vocab_size):
            word = reverse_word_index[word_num]
            embeddings = weights[word_num]
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

    else:
        for word_num in range(0, vocab_size - 1):
            word = tokenizer.decode([word_num])
            embeddings = weights[word_num]
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

    out_v.close()
    out_m.close()


def tokenize_corpus(corpus, num_words=-1):
  # Fit a Tokenizer on the corpus
  if num_words > -1:
    tokenizer = Tokenizer(num_words=num_words)
  else:
    tokenizer = Tokenizer()
  tokenizer.fit_on_texts(corpus)
  return tokenizer

def create_text_corpus(dataset, field):
    # Remove all other punctuation
    dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
    # Make it lowercase
    dataset[field] = dataset[field].str.lower()
    # Make it one long string to split by line
    text = dataset[field].str.cat()
    corpus = text.split('\n')
    # Remove any trailing whitespace
    for l in range(len(corpus)):
        corpus[l] = corpus[l].rstrip()
    # Remove any empty lines
    corpus = [l for l in corpus if l != '']

    return corpus   