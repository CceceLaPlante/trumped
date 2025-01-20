import os
from network import make_model
from preprocessing import getdataset

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

import matplotlib.pyplot as plt

from keras.losses import categorical_crossentropy


def train() :

    os.environ["KERAS_BACKEND"] = "tensorflow"

    train_data, val_data, tokenizer, MAX_LEN, vocab_size = getdataset(vocab_size_minus1=200, user_max_size=200,freq=10)

    tokenizer.save("tokenizer.json")

    X_train = tf.convert_to_tensor(train_data)  
    Y_train = np.zeros(X_train.shape).astype(int)
    Y_train[:, 0:-1] = np.array(X_train[:, 1:])
    Y_train = tf.convert_to_tensor(Y_train)

    X_val = tf.convert_to_tensor(val_data)
    Y_val = np.zeros(X_val.shape).astype(int)
    Y_val[:, 0:-1] = np.array(X_val[:, 1:])
    Y_val = tf.convert_to_tensor(Y_val)


    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_BLOCS = 7
    hidden_dim = 128
    BATCH_SIZE = 64

    data_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    data_train = data_train.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    data_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    data_val = data_val.batch(BATCH_SIZE)


    # ----------------------------------------

    model = make_model(MAX_LEN, vocab_size, EMBED_DIM, NUM_HEADS, NUM_BLOCS, hidden_dim)
    model.summary()



    def scce_with_ls(y, y_hat):
        y = tf.one_hot(tf.cast(y, tf.int32), vocab_size)
        return categorical_crossentropy(y, y_hat, label_smoothing = 0.1)

    # Specify the learning rate here
    learning_rate = 0.001  # Reduced learning rate
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)  # Add gradient clipping
    model.compile(optimizer=optimizer,
                loss=scce_with_ls,
                metrics=["sparse_categorical_accuracy"])


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
            input[0, nb_iter] = argmax_with_temp(output[0, nb_iter - 1], temperature=0.8)
            #input[0, nb_iter] = np.argmax(output[0, nb_iter - 1])
            nb_iter += 1
            if nb_iter == max_iter:
                exit = True

        print("--------------------")
        print(tokenizer.decode(input[0], skip_special_tokens=False) )
        print("--------------------")

    # Add early stopping and learning rate scheduler
    learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
    historic_accuracy = []
    for i in range(10):
        #generate_tweet(model, table)
        print("=========EPOCH " + str(i) + "==========")
        history = model.fit(data_train, epochs=10, verbose=1, validation_data=data_val, callbacks=[ learning_rate_scheduler])
        historic_accuracy.extend(history.history["val_sparse_categorical_accuracy"])
        generate_tweet(model,tokenizer)

        model.save("model_"+str(EMBED_DIM)+"_"+str(NUM_HEADS)+"_"+str(NUM_BLOCS)+"_"+str(hidden_dim)+"_"+str(vocab_size)+"_"+".keras")

    plt.plot(np.array(historic_accuracy).flatten())
    plt.show()

if __name__ == "__main__":
    train()