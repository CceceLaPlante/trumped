from blocs import TransformerBlock, TokenAndPositionEmbedding
import tensorflow as tf
from tensorflow.keras import layers

embed_dim = 32  # Dimension de l'embedding pour chaque mot
num_heads = 2  # Nombre de têtes d'attention


class Net (layers.Layer) :

    def __init__(self, maxlen, vocab_size, embed_dim, num_heads,num_blocs=8) :
        super().__init__()
        self.embeddings = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        
        self.attention_blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_blocs)]
        self.post_processing = layers.Dense(256,activation="relu")
        
        self.softmax = layers.Dense(vocab_size,activation="softmax")
        #dropout
        self.dropout = layers.Dropout
        self.dropout_rate = 0.1
        self.dropout = self.dropout(self.dropout_rate)
        
        
        
    def __call__(self, x) :
        x = self.embeddings(x)
        
        for block in self.attention_blocks :
            x = block(x)
        
        x = self.dropout(x)

        x = self.post_processing(x)
        return self.softmax(x)
    

def make_model (maxlen,vocab_size, embed_dim, num_heads,num_blocs=8) :
    inputs = layers.Input(shape=(maxlen,))
    net = Net(maxlen, vocab_size, embed_dim, num_heads,num_blocs)(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=net)