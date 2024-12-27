
import os
from network import make_model
from preprocessing import getdataset,character_to_number, number_to_character

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

MIN_FREQ = 10

dataset , table, MAX_LEN = getdataset()
table.crop(MIN_FREQ)
print(table.vocab_size)


EMBED_DIM = 128
NUM_HEADS = 3
NUM_BLOCS = 8
hidden_dim = 512
model = make_model(MAX_LEN, table.vocab_size, EMBED_DIM, NUM_HEADS,NUM_BLOCS,hidden_dim)

model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"])

X = tf.convert_to_tensor(dataset)
Y = np.zeros(X.shape).astype(int)
Y[:,0:-1] = np.array(X[:,1:])
Y = tf.convert_to_tensor(Y)
print(X)
print(Y)
#Y = tf.one_hot(Y, depth=table.vocab_size)
# change X to one_hot_vectors

print("pourcentage sur la prédiction de que des 0 : ",np.sum(X == 0)/Y.shape[0]/Y.shape[1]*100)

# example : 
tweet = np.array(X[0])

print("tweet : ",[number_to_character(x,table) for x in tweet])

history = model.fit(X, Y, epochs=30, batch_size=64,verbose=1)
model.save("model.h5")

#text_to_begin_with = "".split(" ")
text_to_begin_with = []
begining = [ table.table[i] for i in text_to_begin_with ]
nb_to_repeat = MAX_LEN - len(begining) -1
input = np.array([[2] + begining + [0] * nb_to_repeat])
exit = False
nb_iter = len(text_to_begin_with)+1
max_iter = MAX_LEN

print( "".join( [table.number_to_character( input[0,i] ) for i in range(len(input[0]))] ) )

while not exit:
    output = model.predict(input)

    input[0,nb_iter] = np.argmax(output[0,nb_iter-1])
    nb_iter += 1
    if nb_iter == max_iter:
        exit = True
    
print( "".join( [table.number_to_character( input[0,i] ) for i in range(len(input[0]))] ) )