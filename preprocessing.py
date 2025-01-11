import csv
import tensorflow as tf
import numpy as np

from tqdm import tqdm

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def load_data () :
    path_to_file = tf.keras.utils.get_file('realdonaltrump.csv', 'https://drive.google.com/uc?export=download&id=1s1isv9TQjGiEr2gG__8bOdBFvQlmepRt')

    tweets = []
    with open(path_to_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append(row['content'])
            
    print("Number of tweets: ", len(tweets))

    
    return tweets
    

def preprocess_tweet(tweet) : 
    filter = '#%*+/;<=>[\\]^_`{|}~\t\n^'
    tweet = tweet.lower()
    tweet = tweet.translate(str.maketrans('', '', filter))
    return tweet
    
    
def getdataset(validation_split=0.2 ,user_max_size = None,vocab_size_minus1=700) :
    tweets = load_data()
    data = []
    
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size_minus1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<SOS>", "<EOS>"],
        )
    
    tokenizer.train_from_iterator(tweets,trainer=trainer)
    tokenizer.add_tokens([" "])
    
    #print(tokenizer.encode(tweets[0]+"<EOS>").ids)
    
    big_ass_list = []
    max_size = 0
    
    for tweet in tweets:
        preprocessed = preprocess_tweet(tweet)
        vec = tokenizer.encode(preprocessed).ids
        
        big_ass_list.extend([2]+vec+[1])
        
        if len(vec) > max_size:
            max_size = len(vec)
        
    max_size = max_size + 2  # Add <sos> and <eos> to the end of the tweet
    
    if user_max_size != None : 
        max_size = user_max_size 
    
    total_size = len(big_ass_list) 
    
    dim1 = total_size//max_size 
    dim2 = max_size 
    
    r = total_size % max_size
    
    print("on print tout ça là oh")
    
    print("dim1 : ", dim1)
    print("dim2", dim2) 
    print(dim1*dim2 + max_size)
    print("reste : ",r)
    
    print("________________")
    
    
    big_ass_list.extend( big_ass_list[0:max_size-r].copy() )
    
    data = np.array( big_ass_list )
    data = data.reshape( (dim1+1,dim2) )
    
    # Split data into training and validation sets
    split_idx = int(dim1 * (1 - validation_split))
    train_data = data[:split_idx,:]
    val_data = data[split_idx:,:]
    
    # Ensure no NaN or infinite values in the dataset
    assert not np.any(np.isnan(train_data)), "Train data contains NaN values"
    assert not np.any(np.isnan(val_data)), "Validation data contains NaN values"
    assert not np.any(np.isinf(train_data)), "Train data contains infinite values"
    assert not np.any(np.isinf(val_data)), "Validation data contains infinite values"
        
    return train_data, val_data, tokenizer, max_size, tokenizer.get_vocab_size()
        
if __name__ == "__main__" :
    train_data, val_data, tokenizer, max_size, vocab_size = getdataset(vocab_size_minus1=500)
    print(train_data[0])
    
    print( tokenizer.decode(train_data[0], skip_special_tokens=False) )
    
    print("vocab _size : ", vocab_size)
    print(tokenizer.get_vocab())