from blocs import TransformerBlock, TokenAndPositionEmbedding
import tensorflow as tf
from tensorflow.keras import layers, models


class Net(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, num_transfo_blocs, hidden_dim):
        super().__init__()
        self.embeddings = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

        self.attention_blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_transfo_blocs)]
        self.post_processing = layers.Dense(hidden_dim, activation="relu")

        self.max_pool1D = layers.MaxPooling1D(maxlen, data_format="channels_last")  # dataformat = "channels_last" pour (batch,steps, features)

        self.softmax = layers.Dense(vocab_size, activation="softmax")

    def __call__(self, x):
        x = self.embeddings(x)

        for block in self.attention_blocks:
            x = block(x)

        x = self.post_processing(x)
        x = self.max_pool1D(x)
        return self.softmax(x)


def make_model(maxlen, vocab_size, embed_dim, num_heads, num_transfo_blocs, hidden_dim):
    inputs = layers.Input(shape=(maxlen,))
    mask = layers.Masking(mask_value=0)(inputs)
    net = Net(maxlen, vocab_size, embed_dim, num_heads, num_transfo_blocs, hidden_dim)(mask)

    return tf.keras.Model(inputs=inputs, outputs=net)