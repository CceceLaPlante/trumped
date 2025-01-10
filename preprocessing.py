import csv
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from keras.layers import TextVectorization


def load_data () :
    path_to_file = tf.keras.utils.get_file('realdonaltrump.csv', 'https://drive.google.com/uc?export=download&id=1s1isv9TQjGiEr2gG__8bOdBFvQlmepRt')

    tweets = []
    with open(path_to_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append(row['content'])
            
    print("Number of tweets: ", len(tweets))

    
    return tweets
    
    
def getdataset(max_size, validation_split=0.2 ,max_vocab_size = 5000):
    tweets = load_data()
    data = []
    
    vectorizer = TextVectorization(max_tokens=max_vocab_size, output_sequence_length=max_size)
    
    text_ds = tf.data.Dataset.from_tensor_slices(tweets).batch(128)
    
    print("fitting...")
    vectorizer.adapt(text_ds)
    vocab = vectorizer.get_vocabulary()
    print("done")
    print("Vocabulary size: ", len(vocab))
        
    
    print("vectorizing...")
    tokenized=  vectorizer(tweets)
    tokenized = tokenized.numpy()
    
    index_to_word = {index: word for index, word in enumerate(vocab)}

    # ok sauf que là tout est pad avec des 0, et on veux pas de ça ! 
    # on vas donc stack les tweets, et les séparer avec des 0
    
    print("stacking...")
    stacked_list = [np.zeros(max_size)]
    idx = 0
    for tweet in tokenized : 
        idx +=1
        for tok in tweet :
            if tok != 0 :
                if idx < max_size :
                    stacked_list[-1][idx] = tok
                    idx += 1
                else : 
                    stacked_list.append(np.zeros(max_size))
                    idx = 0
                    stacked_list[-1][idx] = tok
                    idx += 1
            else : 
                break
    stacked_list = np.array(stacked_list)
    
    vocab[0] = "<PAD>"
    index_to_word[0] = "<PAD>"
    
    train_data= np.array(stacked_list[:int(len(stacked_list) * (1 - validation_split))])
    val_data= np.array(stacked_list[int(len(stacked_list) * (1 - validation_split)):])
    
    
        
    return train_data,val_data, vocab, index_to_word, max_size
    
        
if __name__ == "__main__" :
    train_data, val_data, vocab, index_to_word, max_size = getdataset(300)
    print(train_data[0])
    
    print(  " ".join([index_to_word[i] for i in val_data[0]]) )