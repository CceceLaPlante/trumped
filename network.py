from blocs import TransformerBlock, TokenAndPositionEmbedding
import tensorflow as tf
from tensorflow.keras import layers, models

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class Net(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, num_blocs=1, hidden_layer=64):
        super().__init__()
        self.embeddings = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.attention_blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_blocs)]
        self.batch_norm = layers.BatchNormalization()
        self.post_processing = layers.Dense(hidden_layer, activation="relu")
        self.softmax = layers.Dense(vocab_size, activation="softmax")

    def call(self, inp, mask=None):
        x = self.embeddings(inp)
        for block in self.attention_blocks:
            x = block(x)
        x = self.batch_norm(x)
        x = self.post_processing(x)
        return self.softmax(x)

def make_model(maxlen, vocab_size, embed_dim, num_heads, num_blocs=1, hidden_layer=64):
    inputs = layers.Input(shape=(maxlen,))
    net = Net(maxlen, vocab_size, embed_dim, num_heads, num_blocs, hidden_layer)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=net)