import os
from preprocessing import getdataset,load_data

import nltk
from nltk.translate.gleu_score import sentence_gleu

from tqdm import tqdm

import numpy as np

nltk.download('punkt_tab')
try:
  nltk.data.find('tokenizers/punkt')
except :
  nltk.download('punkt')
  nltk.download('punkt_tab')
  



file = "tweets.txt"

tweets = load_data()

f = open(file, "r")
raw = f.read()
gen_tweets = raw.split("--------------------")

print("Number of tweets: ", len(gen_tweets))
print("Number of tweets: ", len(tweets))


def calculate_bleu(ground_truths, gen_texts , nb_max_tweets = 200):
    
    random_tweets =  np.random.choice(ground_truths, nb_max_tweets)
    
    hyp = nltk.word_tokenize(gen_texts)
    scores = []
    
    max_tweet= ""
    max_score = 0
    for ref in random_tweets : 
        reference_tokens = [nltk.word_tokenize(ref)]
        
        s = sentence_gleu(reference_tokens, hyp)
        scores.append(s)
        if s > max_score : 
            max_score = s
            max_tweet = ref
    
    return max_score, max_tweet


def self_gleu(ground_truths, generated_texts):
    
    scores = []
    for i in tqdm(range(len(generated_texts)),desc="Calculating gleu score"):
        
        score,tw = calculate_bleu(ground_truths, generated_texts[i])
        print("Tweet ", i)
        print("\tGenerated tweet: ", generated_texts[i])
        print("\tGround truth tweet: ", tw)
        print("\tGleu score: ",score )
        
        scores.append(score)
        

        
    print("mean score : ", np.mean(scores))

self_gleu(tweets, gen_tweets)






    


