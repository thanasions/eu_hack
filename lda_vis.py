import gensim
from gensim import corpora
import pyLDAvis.gensim
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import enchant

output = "outputs/"
output += "food.html"

stem = True
english = enchant.Dict("en_GB")
p_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))

start = pd.datetime(2017, 1, 1)
end = pd.datetime(2018, 1, 1)

with open('food_plus_garbage.txt') as handle:
    words_list = handle.read().splitlines()

# N = len(words_list)

print("List length is: " + str(len(words_list)))

texts = []
for i in words_list:
    # doc_set[i] = doc_set[i] + ["_".join(w) for w in ngrams(doc_set[i], 2)]

    # clean and tokenize document string
    raw = i.lower()
    tokens = [raw]
    stopped_tokens = [i for i in tokens if not i in en_stop and english.check(i) and len(i) > 2]
    texts.append(stopped_tokens)

N = len(texts)
print("Nouns List length is: " + str(N))

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
print("Len dict: " + str(len(dictionary)))
# dictionary.filter_extremes(no_below=5, no_above=0.85)
print("Len dict filtered " + str(len(dictionary)))
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, iterations=150)

print(ldamodel)

# pyLDAvis.enable_notebook()
visualisation = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(visualisation, output)
