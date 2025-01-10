import os
from network import make_model
from preprocessing import getdataset

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

def train() :

    os.environ["KERAS_BACKEND"] = "tensorflow"

    MAX_LEN = 100
    train_data, val_data, vocab, index_to_word, MAX_LEN = getdataset(MAX_LEN,max_vocab_size=10000,validation_split=0.1)



    X_train = tf.convert_to_tensor(train_data)
    Y_train = np.zeros(X_train.shape).astype(int)
    Y_train[:, 0:-1] = np.array(X_train[:, 1:])
    Y_train = tf.convert_to_tensor(Y_train)

    X_val = tf.convert_to_tensor(val_data)
    Y_val = np.zeros(X_val.shape).astype(int)
    Y_val[:, 0:-1] = np.array(X_val[:, 1:])
    Y_val = tf.convert_to_tensor(Y_val)


    EMBED_DIM = 1024
    NUM_HEADS = 3
    NUM_BLOCS = 8
    hidden_dim = 512
    BATCH_SIZE = 64

    data_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    data_train = data_train.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    data_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    data_val = data_val.batch(BATCH_SIZE)


    # ----------------------------------------

    model = make_model(MAX_LEN, len(vocab), EMBED_DIM, NUM_HEADS, NUM_BLOCS, hidden_dim)
    model.summary()


    # Specify the learning rate here
    learning_rate = 0.001  # Reduced learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)  # Add gradient clipping

    model.compile(optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"])


    def argmax_with_temp(array, temperature=1.0):
        array = np.log(array) / temperature
        array = np.exp(array)
        array = array / np.sum(array)
        return np.random.choice(len(array), p=array)
        

    def generate_tweet(model,index_to_word):
        
        
        input = np.array([[0] * MAX_LEN])
        exit = False
        nb_iter =  2
        max_iter = MAX_LEN

        while not exit:
            output = model.predict(input, verbose=0)
            input[0, nb_iter] = argmax_with_temp(output[0, nb_iter - 1], temperature=0.95)
            #input[0, nb_iter] = np.argmax(output[0, nb_iter - 1])
            nb_iter += 1
            if nb_iter == max_iter:
                exit = True

        print("--------------------")
        print(" ".join([index_to_word[input[0, i]] for i in range(len(input[0]))]))
        print("--------------------")

    # Add early stopping and learning rate scheduler
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

    for i in range(10):
        #generate_tweet(model, table)
        print("=========EPOCH " + str(i) + "==========")
        generate_tweet(model,index_to_word)
        history = model.fit(data_train, epochs=5, verbose=1, validation_data=data_val, callbacks=[early_stopping, learning_rate_scheduler])
        
        print(f"Epoch {i} history: {history.history}")
        model.save("model.h5")

    model.save("model.h5")

if __name__ == "__main__":
    train()