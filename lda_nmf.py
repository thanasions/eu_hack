# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:58:14 2019

@author: rhydianp
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation
import pickle
import matplotlib.pyplot as plt
with open('users_dict.pickle','rb') as f:
    users_dict = pickle.load(f)
    
users = list(users_dict.keys())
users = [str(x) for x in users]
person_id = []
food = []
image = []
max_df=0.95

no_topics = 8
no_top_words = 20

for i in range(len(users)):
    user_image = list(users_dict[users[i]].keys())
    for j in range(len(user_image)):
        person_id.append(users[i])
        food.append(users_dict[users[i]][user_image[j]])
        image.append(user_image[j])
        

food = [' '.join(x) for x in food]       


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=2,
                                   max_features=no_features, 
                                   stop_words='english',
                                   )
tfidf = tfidf_vectorizer.fit_transform(food)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=max_df, min_df=2,
                                max_features=no_features, 
                                stop_words='english',
                                )
tf = tf_vectorizer.fit_transform(food)
tf_feature_names = tf_vectorizer.get_feature_names()



# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

plt.matshow(nmf.components_)
plt.colorbar()
plt.show()
# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)


print('NMF')
display_topics(nmf, tfidf_feature_names, no_top_words)
print('LDA')
display_topics(lda, tf_feature_names, no_top_words)


