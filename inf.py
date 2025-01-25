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
    zeros = np.where(array == 0)
    array[zeros] = 1e-10
    array = np.log(array) / temperature
    array = np.exp(array)
    s = np.sum(array,axis = -1)
    s = np.expand_dims(s, axis=-1)
    
    array = array / s
    
    return np.array([np.random.choice(len(array[i,:]), p=array[i,:]) for i in range( array.shape[0])])
    

def generate_tweet(model,tokenizer,temp=0.99,nb_tweets=100):
    input = np.array([[0] * MAX_LEN]*nb_tweets)
    exit = False
    nb_iter =  1
    max_iter = MAX_LEN

    while not exit:
        output = model.predict(input, verbose=0)
        input[:,nb_iter] = argmax_with_temp(output[:,nb_iter - 1], temperature=temp)
        #input[0, nb_iter] = np.argmax(output[0, nb_iter - 1])
        nb_iter += 1
        if nb_iter == max_iter:
            exit = True

    
    for i in range(nb_tweets):
        print("--------------------")
        tweet = tokenizer.decode(input[i], skip_special_tokens=False)
        for t in tweet.split("<eos>"):
            print("--------------------")
            print(t.strip("<sos>"))
            
    print("--------------------")
    




tokenizer = Tokenizer.from_file("tokenizer.json")
vs = tokenizer.get_vocab_size()

model =  tf.keras.models.load_model("/home/celeste/trump_gen/trumped_paskassé/trumped/model_256_3_2_256_204_100.h5", custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding, "TransformerBlock": TransformerBlock})

generate_tweet(model,tokenizer,0.8)