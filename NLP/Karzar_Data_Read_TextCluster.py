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

#%%

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





#%%get new text - delete stop words manually

# CleanTextAll = []
# for Text in Data['full_text'].to_list():
#     CleanTextAll.append(tokenize_persian(Text))



#%%LDA
LDA =0
if LDA:

    #data_words = flatten(UniqueText_Lemant)
    import gensim.corpora as corpora# Create Dictionary
    id2word = corpora.Dictionary(CleanTextAll)# Create Corpus
    texts = CleanTextAll# Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]# View
    print(corpus[:1][0][:30])

    print('train')
    import gensim
    from pprint import pprint# number of topics
    num_topics = 5# Build LDA model
   # iteration=100
    from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric
    
    # Set up the callbacks loggers
    perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
    convergence_logger = ConvergenceMetric(logger='shell')
    coherence_cv_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence = 'c_v', texts = texts)
    
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,callbacks=[convergence_logger, perplexity_logger, coherence_cv_logger])# Print the Keyword in the 10 topics
    #lda_model = LdaModel(corpus=corpus, id2word=dictionary, random_state=4583, chunksize=20, num_topics=7, passes=200, iterations=400)
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    
    df = pd.DataFrame.from_dict(lda_model.metrics)



#%%top common words

# tokenized_text = tokenize_persian(Text)
# Counter_text = Counter(tokenized_text)

# Counter_text.most_common(10)

#%%Berttopic
# Data_allt = Data.loc[Data.full_text!="No full text found"]
# TEXTS = Data_allt['full_text'].dropna().to_list()



# from bertopic import BERTopic
# topic_model = BERTopic(embedding_model='HooshvareLab/bert-fa-base-uncased',verbose=True)
# topics, probs = topic_model.fit_transform(TEXTS)

# Res=topic_model.get_topic_info()


# # #%%text topic 1
# Res_df= pd.DataFrame()
# Res_df['Topic']=TEXTS
# Res_df['tweet']=topics
# #Res.to_csv(f'topicmodel_bert-fa-base-uncased_{OnlyFa}.csv')
#%%
# Res_df= pd.DataFrame()
# Res_df['Topic']=topics
# Res_df['tweet']=UniqueText
# Res_df.to_csv(f'Results_Tweet_topicmodel_bert-fa-base-uncased_{OnlyFa}.csv')

#%%bert on title?
Titles = Data['campaign_title'].dropna().to_list()

Titles = [title[17:] for title in Titles] #get rid of the basic initial info



from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import OnlineCountVectorizer
n_nei = 100
#is the number of neighboring sample points used when making the manifold approximation. 
#Increasing this value typically results in a more global view of the embedding structure 
#whilst smaller values result in a more local view. Increasing this value often results in 
#larger clusters being created. 


n_comp = 5

min_dist = 0.1
min_cluster_s=10
nr_topics=100
umap_model = UMAP(n_neighbors=n_nei, n_components=n_comp, min_dist=min_dist, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_s, metric='euclidean',
                        cluster_selection_method='eom', prediction_data=True)
topic_model = BERTopic(embedding_model='HooshvareLab/bert-base-parsbert-uncased',
                             verbose=True)
# umap_model=umap_model,
# hdbscan_model=hdbscan_model,
#  nr_topics=nr_topics,
topics,  probs = topic_model.fit_transform(Titles)

Res_title=topic_model.get_topic_info()

# #%%text topic 1
Res_df_title= pd.DataFrame()
Res_df_title['Topic']=Titles
Res_df_title['tweet']=topics


SAVE = 1
OutPutDir='BertTopic/'
if SAVE:
    Res_df_title.to_csv(OutPutDir+f'BERTTOPIC_ALLList_Title_NO.csv')
    Res_title.to_csv(OutPutDir+f'BERTTOPIC_Topics_Title_NO.csv')
