import csv
import tensorflow as tf


class Table () : 
    
    def __init__ (self) :
        self.table = {"<pad>" : 0, "<eos>" : 1, "<sos>" : 2}
        self.vocab_size = 3
        self.reverse_table = {0:"<pad>",1:"<eos>", 2:"<sos"}
        
    def get_and_add (self, character) :
        if character not in self.table :
            self.table[character] = self.vocab_size
            self.vocab_size += 1
            
            self.reverse_table[self.vocab_size - 1] = character
            
            return self.vocab_size - 1

        else :
            return self.table[character]
        
    def number_to_character (self, number) :
        return self.reverse_table[int(number)]
        

def load_data () :
    path_to_file = tf.keras.utils.get_file('realdonaltrump.csv', 'https://drive.google.com/uc?export=download&id=1s1isv9TQjGiEr2gG__8bOdBFvQlmepRt')

    tweets = []
    with open(path_to_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweets.append(row['content'])
    
    return tweets
    
    
def number_to_character (number,table) :
    return table.number_to_character(number)

def character_to_number (char,table) :
    return table.get_and_add(char)
    

def preprocessing (text, table) :
    """
        we want to clean our tweets from : 
            + urls
            + \n 
            + majuscules 
            
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
            vector.append(character_to_number(char.lower(),table))
            idx += 1

        
        else : 
            new_text += char
            vector.append(character_to_number(char,table))
            idx += 1

            
    
    return vector
             

def padding (vector, max_size) :
    if len(vector) > max_size :
        return vector[:max_size]
    if len(vector) < max_size :
        return  [2]+vector+[1]+[0]*(max_size-len(vector)-2) # on ajoute un <sos> et un <eos> à la fin du tweet
    
def getdataset (token_type = "character", max_size=280) :
    """
        token_type : str, type de tokenisation à utiliser; character ou word, ou token
    """
    # dans un premier temps on implémente que character 
    tweets = load_data()
    data = []
    table = Table()
    
    if token_type == "character" :
        for idx, tweet in enumerate(tweets) :
            vec = preprocessing(tweet,table)
            padded_vec = padding(vec,max_size)
            if padded_vec == None :
                continue
            tensor =padded_vec
            data.append(tensor)
            
    return data,table
            
        
def to_text (vector,table) :
    return "".join([number_to_character(number,table) for number in vector])
        
if __name__ == "__main__" :
    dataset,table = getdataset()
    print("______________________")
    print(to_text(dataset[1],table))
    print(len(dataset))