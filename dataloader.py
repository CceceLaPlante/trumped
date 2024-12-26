import tensorflow as tf
from random import sample

class DataLoader () :
    def __init__ (self,data, batch_size = 32) : 
        self.data = data
        self.batch_size = batch_size
        self.nb_character = 0
        for tweet in data :
            self.nb_character += len(tweet)
                
            
        
    def __len__ (self) :
        return self.nb_character // self.batch_size