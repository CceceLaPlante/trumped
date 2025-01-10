"""
blocs élémentaires pour la constructions du réserau de neurones
"""

import tensorflow as tf
from keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim, value_dim=embed_dim, dropout=0.5)
        self.layernorm1 = layers.LayerNormalization()
        self.ffn = layers.Dense(embed_dim, activation="relu")  # Change activation) to relu for numerical stability
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs):
        attn_output = self.att(inputs, inputs, use_causal_mask=True)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
