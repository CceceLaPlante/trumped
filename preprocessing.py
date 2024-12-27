import csv
import tensorflow as tf

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
    
    def crop (self, min_occurence) : 
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
    splited_text = text.split(" ")
    splited_text = [word.lower() for word in splited_text]  # on met tout en minuscule
    
    new_text = ""
    vector = []
    idx = 0
    while idx < len(splited_text) and len(splited_text) != 0 :
        word = splited_text[idx]
        if word == "\n" :
            idx +=1 
            continue 
        
        elif "http" in word :
            idx += 1
            continue

        else : 
            new_text += word
            vector.append(character_to_number(word,table))
            idx += 1

            
    
    return vector
             

def padding (vector, max_size) :
    if len(vector) >= max_size-2 :
        return [2]+vector[:max_size-2]+[1]

    if len(vector) < max_size :
        return  [2]+vector+[1]+[0]*(max_size-len(vector)-2) # on ajoute un <sos> et un <eos> à la fin du tweet
    
def getdataset () :
    """
        token_type : str, type de tokenisation à utiliser; character ou word, ou token
    """
    # dans un premier temps on implémente que character 
    tweets = load_data()
    data = []
    table = Table()
    max_size = 0
    
    for tweet in tweets :
        vec = preprocessing(tweet,table)
        if len(vec) > max_size :
            max_size = len(vec)
    max_size = max_size + 2 # on ajoute un <sos> et un <eos> à la fin du tweet
    
    for idx, tweet in enumerate(tweets) :
        vec = preprocessing(tweet,table)
        padded_vec = padding(vec,max_size)
        if padded_vec == None :
            continue
        tensor =padded_vec
        data.append(tensor)
            
    print("shocolat")
    return data,table, max_size

class Dataloader () : 
    def __init__ (self,dataset, table,MAX_LEN,batch_size) : 
        self.dataset = dataset
        self.table = table
        self.MAX_LEN = MAX_LEN
        self.batch_size = batch_size    

    def create_batches(self):
        batches = []
        batch_x = []
        batch_y = []

        for tweet in self.dataset:
            for idx, word in enumerate(tweet):
                truncked_tweet = tweet[:idx] + (self.MAX_LEN - idx) * [0]

                # Include padding tokens in the batches
                batch_x.append(truncked_tweet)
                batch_y.append([word])

                if len(batch_x) == self.batch_size:
                    batches.append((batch_x, batch_y))
                    batch_x = []
                    batch_y = []

        if batch_x:
            batches.append((batch_x, batch_y))

        return batches

    def get_generator(self):
        batches = self.create_batches()
        for batch_x, batch_y in batches:
            yield tf.convert_to_tensor(batch_x, dtype=tf.int32), tf.convert_to_tensor(batch_y, dtype=tf.int32)
                    
            
        
def to_text (vector,table) :
    return " ".join([number_to_character(number,table) for number in vector])
        
if __name__ == "__main__" :
    dataset,table,ms = getdataset()
    print("______________________")
    print(to_text(dataset[0],table))
    print(len(dataset))
    print("_____________")
    
    max_len = 0
    for tweet in dataset :
        if len(tweet) > max_len :
            max_len = len(tweet)
    print(max_len)
    
    
    dl = Dataloader(dataset,table,max_len,1)
    gen = dl.get_generator()
    for i in range(20) :
        tt, w = next(gen)
        print(tt,"||", w)
