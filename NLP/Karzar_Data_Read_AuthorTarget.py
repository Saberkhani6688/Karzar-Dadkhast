# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:46:12 2024

@author: User
"""
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import string  
from collections import Counter
def flatten(l):
    return [item for sublist in l for item in sublist]
Puncs = string.punctuation  
Puncs +='،»«'

import os

InputDir = 'Karzar-main/raw_data/'

Data1 = pd.read_excel(InputDir+'Signitures and other Data - 10_5_2024.xlsx')

Data2 = pd.read_excel(InputDir+'Signitures and other Data (5000) - 10_7_2024.xlsx')

Data3 = pd.read_excel(InputDir+'Signitures and other Data (5000) - 10_10_2024.xlsx')

Data4 = pd.read_excel(InputDir+'Signitures and other Data (5000) - 10_22_2024.xlsx')



Data = pd.concat([Data1,Data2,Data3,Data4])
Data = Data.drop_duplicates(subset=['campaign_link'])

#%%text analyze

#analyze text hazm 
from hazm import *
normalizer = Normalizer()
lemmatizer = Lemmatizer()
# tagger = POSTagger( model = 'pos_tagger.model' )
# chunker = Chunker( model= 'chunker.model' )
# parser = DependencyParser (tagger = tagger, lemmatizer = lemmatizer )
#%%tokenize

def tokenize_persian(Text,LemantBol):
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
            if LemantBol:
                Ls_lem.append(lemmatizer.lemmatize(word))
            else:
                Ls_lem.append(word)
        LematWords.append(Ls_lem)
    LematWords_Flat = flatten(LematWords)
    return LematWords_Flat
#%%
Authors = Data.author.to_list()
For= Data.from_info.to_list()


AuthorCounter = Counter(Authors)
ForCounter = Counter(For)
#%%
Values = []
for key, value in AuthorCounter.items():
    Values.append(value)
    
Percentiles = [np.percentile(Values,i) for i in range(10,91,10)]
fix,ax = plt.subplots(1,1,figsize=(10,5))

ax.hist(Values,range=(0,16),bins=100)
ax.grid()
ax.set_title(f'Total Author: {len(Values)} | Percentiles = {Percentiles}')


#%% bert for - us - defining us

BertFor = 1
if BertFor:
    from bertopic import BERTopic
    
    #topic model on the "for"
    ForWords = list(ForCounter.keys())
    
    
    topic_model = BERTopic(embedding_model='HooshvareLab/bert-base-parsbert-uncased',
                                 verbose=True)
    
    topics,  probs = topic_model.fit_transform(ForWords)
    
    Res_ForWords=topic_model.get_topic_info()
    
    # #%%text topic 1
    Res_df_ForWords= pd.DataFrame()
    Res_df_ForWords['Topic']=ForWords
    Res_df_ForWords['For_Info']=topics
    
    SAVE = 1
    OutPutDir='BertTopic/'
    if SAVE:
        Res_df_ForWords.to_csv(OutPutDir+f'BERTTOPIC_ALLList_ForWords_NO.csv')
        Res_ForWords.to_csv(OutPutDir+f'BERTTOPIC_Topics_ForWords_NO.csv')
    
#%%target
Texts = Data.full_text.to_list()

#%%intors - getting the "hello"
import re
Intros = []
word_to_match = "سلام"
word_to_match2 = "درود"
word_to_match3 = "احترام"

pattern = r"\b" + re.escape(word_to_match) + r"\b"
pattern2 = r"\b" + re.escape(word_to_match2) + r"\b"
pattern3 = r"\b" + re.escape(word_to_match3) + r"\b"
SkippedText = []
for text in Texts:
        if type(text)==float:
            Intros.append('Skip')
            continue
        if len(text)<45:
            Intros.append('Skip')
            continue
        match = re.search(pattern, text)
        match2 = re.search(pattern2, text)
        match3 = re.search(pattern3, text)

        if match!=None:

            Intros.append(text[:match.span()[0]])
        elif match2!=None:
            Intros.append(text[:match2.span()[0]])
        elif match3!=None:
            Intros.append(text[:match3.span()[0]])
        else:
            Intros.append('Skip')
            SkippedText.append(text)

test = Counter(Intros)
test.most_common(10)
    
#%% getting the first words maybe?
InitialText = []
for text in Texts:
        if type(text)==float:
            Intros.append('Skip')
            continue
        if len(text)<45:
            Intros.append('Skip')
            continue
        InitialText.append(text[:100])
        

LemantBol =0 

CleanInitialTextAll = []
for Text in InitialText:
    CleanInitialTextAll.append(tokenize_persian(Text,LemantBol))
    #%%
CleanInitialFlat = flatten(CleanInitialTextAll)
CounterClean = Counter(CleanInitialFlat)
CounterClean.most_common(50)


#%%maybe bert it?

BertTarget = 1
if BertTarget:
    from bertopic import BERTopic
    
    #topic model on the "for"
    
    
    topic_model = BERTopic(embedding_model='HooshvareLab/bert-base-parsbert-uncased',
                                 verbose=True)
    
    topics,  probs = topic_model.fit_transform(InitialText)
    
    Res_Target=topic_model.get_topic_info()
    
    # #%%text topic 1
    Res_df_Target= pd.DataFrame()
    Res_df_Target['Target']=InitialText
    Res_df_Target['Topics']=topics
    
    SAVE = 1
    OutPutDir='BertTopic/'
    if SAVE:
        Res_df_Target.to_csv(OutPutDir+f'BERTTOPIC_ALLList_Target_NO.csv')
        Res_Target.to_csv(OutPutDir+f'BERTTOPIC_Topics_Targets_NO.csv')