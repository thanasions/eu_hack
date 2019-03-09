from gensim.models import KeyedVectors
from tqdm import tqdm
import os
import zipfile


class Tags(object):
    def __init__(self, file_name='models/glove/glove2vec.6B.50d.txt', threshold=None):
        self.__stopwords = ['food', 'meal', 'dinner', 'breakfast', 'lunch', 'snack']
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
        self.__model = KeyedVectors.load_word2vec_format(self.__file_name)

    def calculate(self, image_labels):
        returned_tags = {
            'fruit_veg': 0,
            'starchy': 0,
            'protein': 0,
            'fat_sugars': 0,
            'dairy': 0
        }

        for image_label in image_labels:
            for img_label_term in image_label.split():
                if img_label_term.lower() not in self.__stopwords:
                    for tag, search_terms in self.__predefined_tags.items():
                        for search_term in search_terms:
                            try:
                                similarity_score = self.__model.similarity(img_label_term.lower(), search_term)
                                returned_tags[tag] = max(similarity_score, returned_tags[tag])
                            except:
                                continue
        return returned_tags


