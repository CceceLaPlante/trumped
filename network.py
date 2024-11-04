from .blocs import TransformerBlock, TokenAndPositionEmbedding
import tensorflow as tf
from tensorflow.keras import layers

embed_dim = 32  # Dimension de l'embedding pour chaque mot
num_heads = 2  # Nombre de têtes d'attention

class Net (layers.Layer) :

    def __init__(self, maxlen, vocab_size, embed_dim, num_heads) :
        super().__init__()
        self.embeddings = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.att = TransformerBlock(embed_dim, num_heads)
        self.proj = layers.GlobalAveragePooling1D()
        self.post_processing = layers.Dense(20,activation="relu")
        self.softmax = layers.Dense(vocab_size,activation="softmax")
        
    def __call__(self, x) :
        x = self.embeddings(x)
        x = self.att(x)
        x = self.proj(x)
        x = self.post_processing(x)
        return self.softmax(x)
    

def make_model (maxlen,vocab_size, embed_dim, num_heads) :
    inputs = layers.Input(shape=(maxlen,))
    net = Net(maxlen, vocab_size, embed_dim, num_heads)(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=net)


    
        