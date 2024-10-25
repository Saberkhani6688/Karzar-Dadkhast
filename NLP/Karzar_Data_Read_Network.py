# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:46:12 2024

@author: User
"""

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

LemantBol = 1
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
SIGNATURES = []
for idx, row in Data.iterrows():
    
    sign = row.signatures
    sign = sign.replace(']','')
    sign = sign.replace('[','')
    sign = sign.split(',')
    sign = [s[2:-1] for s in sign]
    SIGNATURES.append(sign)
    
    
#%%
def Jaccard_index_list(data1,data2):
    jaccard_results = []
    for i in range (len(data1)):
        lst1 = tokenizer(data1.iloc[i])
        lst2 = tokenizer(data2.iloc[i])
        result = len(intersection(lst1, lst2))/ len(union(lst1, lst2))  
        jaccard_results.append(result)
    
    return jaccard_results

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def Jaccard_index(lst1,lst2):
    return len(intersection(lst1, lst2))/ len(union(lst1, lst2))
#%%
import time
StartTime= time.time()

AllPossiblePairs = len(Data)*len(Data)/2
c = 0 
JaccardIndexSign  = np.zeros((len(Data),len(Data)))
for i in range(len(Data)):
    for j in range(len(Data)):
        if j<=i:
            continue
        JaccardIndexSign[i,j] = Jaccard_index(SIGNATURES[i],SIGNATURES[j])
        c+=1
        if c%(int(AllPossiblePairs/100))==0:
            print(time.time()-StartTime)
            StartTime= time.time()
            print(f'{c}/{AllPossiblePairs}')
            
            
            
#%%
from matplotlib import pyplot as plt

plt.hist(JaccardIndexSign)

#np.save('JaccardIndexAll',JaccardIndexSign)
#%%jaccard values
JaCval = []
for i in range(len(Data)):
    for j in range(len(Data)):
        if j<=i:
            continue
        JaCval.append(JaccardIndexSign[i,j])
np.mean(JaCval)
#%%
import networkx as nx
G = nx.from_numpy_matrix(JaccardIndexSign)
G.edges(data=True)
#%%cent
deg_centrality = nx.degree_centrality(G)
centrality = np.fromiter(deg_centrality.values(), float)

LabelsNodes = []
plt.hist(centrality)

#%%
Df_network = pd.DataFrame()
Df_network['title']= Data.campaign_title
Df_network['cent'] = centrality
Df_network['progress_percentage'] = Data.progress_percentage
Df_network.to_csv('Df_Network.csv')
#%%
# pos = nx.kamada_kawai_layout(G,scale = 10)
 
# nx.draw(G,pos=pos,node_color=centrality, node_size=centrality*2e3)#,labels =UserID_df_dict_network)


#%%hist of preveleance
SIGNATURES_f = flatten(SIGNATURES)
from collections import Counter
Counter_sign = Counter(SIGNATURES_f)
print(Counter_sign.most_common(100))

Values = []
for key, value in Counter_sign.items():
    Values.append(value)
#%%
Percentiles = [np.percentile(Values,i) for i in range(10,91,10)]
fix,ax = plt.subplots(1,1,figsize=(10,5))

ax.hist(Values,range=(0,2500),bins=100)
ax.grid()
ax.set_title(f'Total User: {len(Values)} | Percentiles = {Percentiles}')
