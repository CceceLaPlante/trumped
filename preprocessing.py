import csv
import tensorflow as tf


def load_data () :
    path_to_file = tf.keras.utils.get_file('realdonaltrump.csv', 'https://drive.google.com/uc?export=download&id=1s1isv9TQjGiEr2gG__8bOdBFvQlmepRt')

    tweets = []
    with open(path_to_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append(row['content'])
    
    return tweets
    
    
def number_to_character (number) :
    return chr(number)

def preprocessing (text) :
    """
        we want to clean our tweets from : 
            + urls
            + \n 
            + majuscules *
            
        finaly we change our character to numbers
        
    """
    
    new_text = ""
    vector = []
    idx = 0
    while idx < len(text) and len(text) != 0 :
        char = text[idx]
        if char == "\n" :
            idx +=1 
            continue 
        
        if char == "h" and text[idx:idx+4] == "http" :
            while idx < len(text) and text[idx] != " " :
                idx += 1
            continue
        
        if char.isupper() :
            new_text += char.lower()
            vector.append(ord(char.lower()))
            idx += 1

        
        else : 
            new_text += char
            vector.append(ord(char))
            idx += 1

            
    
    return vector
             

def padding (vector, max_size) :
    if len(vector) > max_size :
        return vector[:max_size]
    if len(vector) < max_size :
        return vector + [0]*(max_size-len(vector))
    
def getdataset (token_type = "character", max_size=280) :
    """
        token_type : str, type de tokenisation à utiliser; character ou word, ou token
    """
    # dans un premier temps on implémente que character 
    tweets = load_data()
    data = []
    
    if token_type == "character" :
        for idx, tweet in enumerate(tweets) :
            vec = preprocessing(tweet)
            padded_vec = padding(vec,max_size)
            if padded_vec == None :
                continue
            tensor = tf.convert_to_tensor(padded_vec)
            data.append(tensor)
            
    return data
            
        
def to_text (vector) :
    return "".join([number_to_character(number) for number in vector])
        
        
        
if __name__ == "__main__" :
    dataset = getdataset()
    print("______________________")
    print(to_text(dataset[1]))
    print(len(dataset))