import os
from network import make_model
from preprocessing import getdataset

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from blocs import TokenAndPositionEmbedding, TransformerBlock

from tokenizers import Tokenizer

MAX_LEN=100


import matplotlib.pyplot as plt

def argmax_with_temp(array, temperature=1.0):
    array = np.log(array) / temperature
    array = np.exp(array)
    array = array / np.sum(array)
    return np.random.choice(len(array), p=array)

def generate_tweet(model,tokenizer):
    input = np.array([[0] * MAX_LEN])
    exit = False
    nb_iter =  2
    max_iter = MAX_LEN

    while not exit:
        output = model.predict(input, verbose=0)
        input[0, nb_iter] = argmax_with_temp(output[0, nb_iter - 1], temperature=0.99)
        #input[0, nb_iter] = np.argmax(output[0, nb_iter - 1])
        nb_iter += 1
        if nb_iter == max_iter:
            exit = True

    print("--------------------")
    print(tokenizer.decode(input[0], skip_special_tokens=False) )
    print("--------------------")




tokenizer = Tokenizer.from_file("tokenizer.json")
vs = tokenizer.get_vocab_size()

model =  tf.keras.models.load_model("/home/celeste/trump_gen/trumped/model_256_4_6_256_231_.h5", custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding, "TransformerBlock": TransformerBlock})

generate_tweet(model,tokenizer)
generate_tweet(model,tokenizer)
generate_tweet(model,tokenizer)
generate_tweet(model,tokenizer)