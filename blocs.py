"""
blocs élémentaires pour la constructions du réserau de neurones
"""

import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    # embed_dim désigne la dimension des embeddings maintenus à travers les différentes couches,
    # et num_heads le nombre de têtes de la couche d'attention.
    # DANS CETTE FONCTION, ON NE FAIT QUE DEFINIR LES COUCHES
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # Définition des différentes couches qui composent le bloc
        # Couche d'attention
        self.att = layers.MultiHeadAttention(num_heads, embed_dim,value_dim=embed_dim)
        # Première couche de Layer Normalization
        self.layernorm1 = layers.LayerNormalization()
        # Couche Dense (Feed-Forward)
        self.ffn = layers.Dense(embed_dim, activation="softmax")
        # Deuxième couche de normalisation
        self.layernorm2 =layers.LayerNormalization()

    # DANS CETTE FONCTION, ON APPELLE EXPLICITEMENT LES COUCHES DEFINIES DANS __init__
    # ON PROPAGE DONC LES ENTREES inputs A TRAVERS LES DIFFERENTES COUCHES POUR OBTENIR
    # LA SORTIE
    def call(self, inputs):
        # Application des couches successives aux entrées
        x = self.att(inputs,inputs, use_causal_mask=True)
        y = self.layernorm1(x+ inputs)
        z = self.ffn(y)
        x = self.layernorm2(z+y)
        return x

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        # Définition des différentes couches qui composent le bloc Embedding
        # Embedding de mot
        self.token_emb = layers.Embedding(vocab_size,embed_dim)
        # Embedding de position
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        # Calcul de l'embedding à partir de l'entrée x
        # ATTENTION : UTILISER UNIQUEMENT DES FONCTIONS TF POUR CETTE PARTIE
        # Récupération de la longueur de la séquence
        # on a un vecteur de taille (1,maxlen)
        maxlen = tf.shape(x)[-1]
        # Création d'un vecteur [0, 1, ..., maxlen] des positions associées aux
        # mots de la séquence (fonction tf.range)
        positions = tf.range( maxlen )
        # Calcul des embeddings de position
        positions_emb = self.pos_emb(positions)
        # Calcul des embeddings de mot
        words_emb = self.token_emb(x)
        return positions_emb + words_emb