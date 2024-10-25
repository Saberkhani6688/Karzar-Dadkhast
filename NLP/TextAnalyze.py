import pandas as pd
import numpy as np
import string  
from collections import Counter
def flatten(l):
    return [item for sublist in l for item in sublist]
Puncs = string.punctuation  
Puncs +='،»«'
f = open('text.txt', 'r',encoding="utf8")


Text = f.read()


f.close()

#%%analyze text hazm 
from hazm import *
normalizer = Normalizer()
lemmatizer = Lemmatizer()
# tagger = POSTagger( model = 'pos_tagger.model' )
# chunker = Chunker( model= 'chunker.model' )
# parser = DependencyParser (tagger = tagger, lemmatizer = lemmatizer )
#%%tokenize
NorText = normalizer.normalize (Text)
Sentences = sent_tokenize(NorText)
Words = []
for sent in Sentences:
    Words.append(word_tokenize(sent))
    
##lemat
LematWords = []
for ls_w in Words:
    Ls_lem = []
    for word in ls_w:
        #stop word
        if word in stopwords_list() or word in Puncs:
            continue
        Ls_lem.append(lemmatizer.lemmatize(word))
    LematWords.append(Ls_lem)
    
#%%top common words
LematWords_Flat = flatten(LematWords)
Counter_text = Counter(LematWords_Flat)

Counter_text.most_common(10)

#%%NER 

###Beheshti-NER
