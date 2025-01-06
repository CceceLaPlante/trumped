import csv
import tensorflow as tf
import numpy as np

class Table () : 
    
    def __init__ (self) :
        self.table = {"<pad>" : 0, "<eos>" : 1, "<sos>" : 2}
        self.vocab_size = 3
        self.reverse_table = {0:"<pad>",1:"<eos>", 2:"<sos>"}
        self.frequency = {}
        
    def get_and_add (self, character) :
        if character not in self.table :
            self.table[character] = self.vocab_size
            self.vocab_size += 1
            self.frequency[character] = 1
            
            self.reverse_table[self.vocab_size - 1] = character
            
            return self.vocab_size - 1

        else :
            self.frequency[character] += 1
            return self.table[character]
        
    def number_to_character (self, number) :
        return self.reverse_table[int(number)]
    
    def crop (self, min_occurence, vectors) : 
        
        new_table = {"<pad>" : 0, "<eos>" : 1, "<sos>" : 2}
        new_reverse_table = {0:"<pad>",1:"<eos>", 2:"<sos>"}
        new_vocab_size = 3
        new_frequency = {}
        
        for key in self.table :
            if key == "<pad>" or key == "<eos>" or key == "<sos>" :
                continue
            if self.frequency[key] >= min_occurence :
                new_table[key] = new_vocab_size
                new_reverse_table[new_vocab_size] = key
                new_vocab_size += 1
                new_frequency[key] = self.frequency[key]
        
        self.table = new_table
        self.reverse_table = new_reverse_table
        self.vocab_size = new_vocab_size
        self.frequency = new_frequency
    
        
        

def load_data () :
    path_to_file = tf.keras.utils.get_file('realdonaltrump.csv', 'https://drive.google.com/uc?export=download&id=1s1isv9TQjGiEr2gG__8bOdBFvQlmepRt')

    tweets = []
    with open(path_to_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append(row['content'])
            
    print("Number of tweets: ", len(tweets))

    
    return tweets
    
    
def number_to_character (number,table) :
    return table.number_to_character(number)

def character_to_number (char,table) :
    return table.get_and_add(char)


def preprocessing(text, table):
    splited_text = text
    splited_text = [word.lower() for word in splited_text]  # Convert to lowercase
    
    new_text = ""
    vector = []
    idx = 0
    while idx < len(splited_text) and len(splited_text) != 0:
        word = splited_text[idx]
        if word == "\n" or "http" in word:
            idx += 1
            continue
        else:
            new_text += word
            vector.append(character_to_number(word, table))
            idx += 1
            
    return vector

def padding(vector, max_size):
    if len(vector) >= max_size - 2:
        return [2] + vector[:max_size - 2] + [1]
    if len(vector) < max_size:
        return [2] + vector + [1] + [0] * (max_size - len(vector) - 2)  # Add <sos> and <eos> tokens
    
def getdataset(validation_split=0.2, user_max_size = None ):
    tweets = load_data()
    data = []
    table = Table()
    max_size = 0
    
    big_ass_list = []
    
    for tweet in tweets:
        vec = preprocessing(tweet, table)
        
        big_ass_list.extend([2]+vec+[1])
        
        if len(vec) > max_size:
            max_size = len(vec)
        
    max_size = max_size + 2  # Add <sos> and <eos> to the end of the tweet
    """
    for idx, tweet in enumerate(tweets):
        vec = preprocessing(tweet, table)
        padded_vec = padding(vec, max_size)
        if padded_vec is None:
            continue
        tensor = padded_vec
        data.append(tensor)"""
        
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
    
    return train_data, val_data, table, max_size
            
        
def to_text (vector,table) :
    return "".join([number_to_character(number,table) for number in vector])
        
if __name__ == "__main__" :
    train_data, val_data, table, max_size = getdataset()
    print("______________________")
    print(to_text(train_data[1],table))
    print(len(train_data))