from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import zipfile
from nltk import word_tokenize, PorterStemmer
from scipy import spatial


class Tags(object):
    def __init__(self, file_name='models/glove/glove2vec.6B.50d.txt', doc=False,threshold=None):
        ps = PorterStemmer()
        stopwords = ['cuisine','food', 'meal', 'dinner', 'breakfast', 'lunch', 'snack', 'dish', 'ingredient', 'recipe', 'produce']
        self.__stopwords = [ps.stem(x) for x in stopwords]
        self.__predefined_tags = {
            'fruit_veg': ['fruit', 'vegetables'],
            'starchy': ['bread', 'rice', 'pasta', 'potatoes'],
            'protein': ['meat', 'egg', 'fish', 'beans', 'legumes'],
            'fat_sugars': ['cake', 'junk'],
            'dairy': ['milk', 'cheese', 'yoghurt']
        }

        self.__file_name = file_name

        if not os.path.isfile(file_name):
            with zipfile.ZipFile(file_name+".zip","r") as zip_ref:
                zip_ref.extractall("models/glove/")
        self.__model = KeyedVectors.load_word2vec_format(self.__file_name) if not doc else Doc2Vec.load(file_name)

    def traindoc2vec(self, data):

        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
        max_epochs = 200
        vec_size = 100
        alpha = 0.025

        model = Doc2Vec(size=vec_size,
                        alpha=alpha,
                        min_alpha=0.00025,
                        min_count=1,
                        dm=1)

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        model.save("models/d2v.model")

    def calculate_word(self, image_labels):
        returned_tags = {
            'fruit_veg': 0,
            'starchy': 0,
            'protein': 0,
            'fat_sugars': 0,
            'dairy': 0
        }
        ps = PorterStemmer()
        for image_label in image_labels:
            for img_label_term in image_label.split():
                if ps.stem(img_label_term.lower()) not in self.__stopwords:
                    for tag, search_terms in self.__predefined_tags.items():
                        for search_term in search_terms:
                            try:
                                similarity_score = self.__model.similarity(img_label_term.lower(), search_term)
                                returned_tags[tag] = max(similarity_score, returned_tags[tag])
                            except:
                                continue
        return returned_tags

    def calculate_doc(self, image_labels):
        returned_tags = {
            'fruit_veg': 0,
            'starchy': 0,
            'protein': 0,
            'fat_sugars': 0,
            'dairy': 0
        }
        labels = [x.split() for x in image_labels]
        labels = [j.lower() for sub in labels for j in sub]
        vec1 = self.__model.infer_vector(labels)

        for tag, search_terms in self.__predefined_tags.items():
            vec2 = self.__model.infer_vector(search_terms)
            returned_tags[tag] = spatial.distance.cosine(vec1, vec2)
        return returned_tags


