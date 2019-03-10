# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:33:35 2019

@author: rhydianp
"""

from gensim import corpora, models, similarities 
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.tag import pos_tag
import re
import nltk
import pickle
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram


with open('users_dict.pickle','rb') as f:
    users_dict = pickle.load(f)
    

stemmer = SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words('english')

users = list(users_dict.keys())
users = [str(x) for x in users]
person_id = []
food = []
image =[]

for i in range(len(users)):
    user_image = list(users_dict[users[i]].keys())
    for j in range(len(user_image)):
        person_id.append(users[i])#*len(users_dict[users[i]][user_image[j]]))
        food.append(users_dict[users[i]][user_image[j]])
        image.append(user_image[j])
        

food = [' '.join(x) for x in food]
with open('food_plus_garbage.txt') as handle:
    food = handle.read().splitlines()

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns


#remove proper names
preprocess = [strip_proppers(x) for x in food]

#tokenize
tokenized_text = [tokenize_only(text) for text in food]

#remove stop words
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]


#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
#dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaModel(corpus, num_topics=10, 
                            id2word=dictionary, 
                            update_every=5, 
                            chunksize=10000, 
                            passes=50)


topics_matrix = lda.show_topics(formatted=False)
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(lda.get_topics())



# =============================================================================
# MDS()
# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
# 
# pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
# 
# xs, ys = pos[:, 0], pos[:, 1]
# 
# 
# plt.scatter(x = xs, y = ys)
# plt.show()
# =============================================================================


linkage_matrix = ward(dist)
plt.clf()
plt.figure(figsize=(40,20))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('person Id')
plt.ylabel('Distance')
dendrogram(
        linkage_matrix,
        leaf_rotation=90.,
        leaf_font_size=12.)
plt.savefig('ward_dendrogram.png',dpi = 200)
plt.show()

# print(linkage_matrix)
# print(lda.get_topics())
