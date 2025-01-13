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

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def word_or_unknown (word,word_freq,freq=3) : 
    if word_freq[word] > freq or is_ascii(word) : 
        return word
    else :
        return "<UNK>"
    
def remove_non_freq_words (tweets, freq=3) : 
    word_freq = {" ":freq+1}
    for tweet in tweets:
        for word in tweet:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    tweets = [''.join([word_or_unknown(word,word_freq,freq) for word in tweet]) for tweet in tweets]
    
    return tweets

def preprocess_tweet(tweet) : 
    filter = '#%*+/;=[\\]^_`{|}~\t\n^*§¨`~'
    tweet = tweet.lower()
    
    new_tweet = ""
    for char in tweet:
        if char not in filter:
            new_tweet += char
        else:
            new_tweet += ""
    
    return new_tweet
    
    
def getdataset(validation_split=0.2 ,user_max_size = None,vocab_size_minus1=700,freq=3) :
    tweets = remove_non_freq_words(load_data(), freq=freq)
    preprocessed_tweet = [preprocess_tweet(tweet) for tweet in tweets]
    
    print("nb chars : ")
    chars = []
    for tweet in preprocessed_tweet:
        for char in tweet:
            if char not in chars:
                chars.append(char)
    print(len(chars))
    data = []
    
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size_minus1,
        special_tokens=["<sos>", "<eos>","<unk>"],
        )
    
    tokenizer.train_from_iterator(preprocessed_tweet,trainer=trainer)
    tokenizer.add_tokens([" "])
    
    #print(tokenizer.encode(tweets[0]+"<EOS>").ids)
    
    big_ass_list = []
    max_size = 0
    
    for tweet in preprocessed_tweet:
        tw = "<sos>"+tweet+"<eos>"
        vec = tokenizer.encode(tw).ids
        
        big_ass_list.extend(vec)
        
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
    train_data, val_data, tokenizer, max_size, vocab_size = getdataset(vocab_size_minus1=200,freq=20)
    print(train_data[0])
    
    print( tokenizer.decode(train_data[0], skip_special_tokens=False) )
    
    print("vocab _size : ", vocab_size)
    print(tokenizer.get_vocab())