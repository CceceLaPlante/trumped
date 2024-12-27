import os
from network import make_model
from preprocessing import getdataset, character_to_number, number_to_character

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

MIN_FREQ = 20

train_data, val_data, table, MAX_LEN = getdataset()
table.crop(MIN_FREQ)
print(table.vocab_size)

# Check for NaN values in the input data
assert not np.any(np.isnan(train_data)), "Train data contains NaN values"
assert not np.any(np.isnan(val_data)), "Validation data contains NaN values"

EMBED_DIM = 64
NUM_HEADS = 3
NUM_BLOCS = 5
hidden_dim = 256
BATCH_SIZE = 64

model = make_model(MAX_LEN, table.vocab_size, EMBED_DIM, NUM_HEADS, NUM_BLOCS, hidden_dim)
model.summary()

def custom_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss *= mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())  # Add epsilon to prevent division by zero

# Specify the learning rate here
learning_rate = 0.00001  # Reduced learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)  # Use RMSprop optimizer

model.compile(optimizer=optimizer,
              loss=custom_loss,
              metrics=["sparse_categorical_accuracy"])

X_train = tf.convert_to_tensor(train_data)
Y_train = np.zeros(X_train.shape).astype(int)
Y_train[:, 0:-1] = np.array(X_train[:, 1:])
Y_train = tf.convert_to_tensor(Y_train)

X_val = tf.convert_to_tensor(val_data)
Y_val = np.zeros(X_val.shape).astype(int)
Y_val[:, 0:-1] = np.array(X_val[:, 1:])
Y_val = tf.convert_to_tensor(Y_val)

data_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
data_train = data_train.shuffle(buffer_size=1024).batch(BATCH_SIZE)

data_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
data_val = data_val.batch(BATCH_SIZE)

#check for NaN values in the input data
assert not np.any(np.isnan(train_data)), "Train data contains NaN values"
assert not np.any(np.isnan(val_data)), "Validation data contains NaN values"


class CheckNaNCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if np.isnan(logs['loss']):
            print(f"NaN loss detected in batch {batch}")
            self.model.stop_training = True
            

def argmax_with_temp(array, temperature=1.0):
    array = np.log(array) / temperature
    array = np.exp(array)
    array = array / np.sum(array)
    return np.random.choice(len(array), p=array)
      

def generate_tweet(model, table):
    text_to_begin_with = "make america".split(" ")
    text_to_begin_with = []
    begining = [table.table[i] for i in text_to_begin_with]
    nb_to_repeat = MAX_LEN - len(begining) - 1
    input = np.array([[2] + begining + [0] * nb_to_repeat])
    exit = False
    nb_iter = len(text_to_begin_with) + 1
    max_iter = MAX_LEN

    while not exit:
        output = model.predict(input, verbose=0)
        #input[0, nb_iter] = argmax_with_temp(output[0, nb_iter - 1], temperature=1)
        input[0, nb_iter] = np.argmax(output[0, nb_iter - 1])
        nb_iter += 1
        if nb_iter == max_iter:
            exit = True

    print("--------------------")
    print(" ".join([table.number_to_character(input[0, i]) for i in range(len(input[0]))]))
    print("--------------------")

# Add early stopping and learning rate scheduler
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
check_nan = CheckNaNCallback()

for i in range(10):
    generate_tweet(model, table)
    print("=========EPOCH " + str(i) + "==========")
    model.fit(data_train, epochs=5, batch_size=BATCH_SIZE, verbose=1, validation_data=data_val, callbacks=[early_stopping, check_nan,learning_rate_scheduler])
    model.save("model.h5")

model.save("model.h5")