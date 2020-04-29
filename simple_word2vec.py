#!/usr/bin/env python
# coding: utf-8

# # SET DATA

# In[ ]:


# Set of documents
docs=["the house had a tiny little mouse",
      "the cat saw the mouse",
      "the mouse ran away from the house",
      "the cat finally ate the mouse",
      "the end of the mouse story"
     ]

# Corpus of distinct word
corpus = ["the", "house", "had", "a", "tiny", "little", "mouse", "cat", "saw",
          "mouse", "ran", "away", "away", "from", "finally", "ate", "the", "end", "of", "story"]


# # DATA PREPARATION

# In[ ]:


from collections import defaultdict

# tokenization by word
data = [doc.split() for doc in docs]

words_count = defaultdict(int)
for row in data:
    for word in row:
        words_count[word] += 1

# number of unique words
v_count = len(words_count.keys()) 

# list of unique words
words_list = list(words_count.keys()) 

# word:index
word_index = dict((word, i) for i, word in enumerate(words_list)) 

#index:word
index_word = dict(map(lambda item: (item[1], item[0]), word_index.items())) 


# # COUNT VECTOR

# In[ ]:


def word2countvector(data):
    words_encoded = dict()
    # Encode all words in our corpus
    for word in words_list:
        word_vec = [0 for i in range(len(data))]
        
        for i in range(len(data)):
            word_vec[i] = data[i].count(word)

        words_encoded[word] = word_vec
        
    # Encode all our dataset
    data_countvector_encoded = []
    for doc in data:
        countvector_doc = []
        for word in doc:
            countvector_doc.append(words_encoded[word])
        data_countvector_encoded.append(countvector_doc)
    return words_encoded, data_countvector_encoded


# # ONE HOT ENCODING

# In[ ]:


def word2onehot(data):
    
    data_onehot_encoded = []
    
    for doc in data:
        onehot_doc = []
        for word in doc:
            word_vec = [0 for i in range(v_count)]

            index = word_index[word]

            word_vec[index] = 1
            onehot_doc.append(word_vec)
        data_onehot_encoded.append(onehot_doc)
    return data_onehot_encoded


# # TF-IDF

# In[28]:


from math import log
def word2tfidf(data):
    def presence_in_document(word):
        presence = 0
        for doc in data:
            presence += 1 if doc.count(word) != 0 else 0
        return presence
    def replace(doc, x, y):
        return list(map((lambda item: y if item == x else item), doc))
    
    data_tfidf_encoded = []
    for doc in data:
        tfidf_doc = doc
        for word in doc:
            if isinstance(word, str):
                tf = doc.count(word)/len(doc)
                idf = log((len(data)/presence_in_document(word)), 10)

                numeric_word = tf*idf
                tfidf_doc = replace(tfidf_doc, word, numeric_word)
        data_tfidf_encoded.append(tfidf_doc)
    return data_tfidf_encoded


# # VECTORISATION

# ## Vectorisation using CountVector method

# In[30]:


words_encoded, data_countvector_encoded = word2countvector(data)
data_countvector_encoded


# ## Vectorisation using One Hot Encoding method

# In[32]:


data_onehot_encoded = word2onehot(data)
data_onehot_encoded


# ## Vectorisation using TF-IDF Method

# In[33]:


data_tfidf_encoded = word2tfidf(data)
data_tfidf_encoded

