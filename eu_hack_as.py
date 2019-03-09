import io
import os
import json

# Imports the Google Cloud client library
import pickle
import time

from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account

PATH = os.getcwd()
path_to_json = os.path.join(PATH,'NY Taxis-a3cc565a66b6.json')
path_to_self_reports = os.path.join(PATH, 'dataset', 'big0', 'self_reports')

credentials = service_account.Credentials.from_service_account_file(path_to_json)
client = vision.ImageAnnotatorClient(credentials=credentials)


def __get_dict(user_dir):
        meals={}
        if 'meals' in os.listdir(user_dir):
            for file in os.listdir(os.path.join(user_dir, 'meals')):
                if file.endswith('jpg'):
                    with io.open(os.path.join(user_dir, 'meals', file), 'rb') as image_file:
                        content = image_file.read()
                        image = types.Image(content=content)
                        # Performs label detection on the image file
                        response = client.label_detection(image=image)
                        labels = response.label_annotations
                        meal_number = file.split(".")[0]
                        meals[meal_number] = [ l.description for l in labels ]
        return meals

def __get_all_user_dicts():
    users_dict={}
    for d in os.listdir(path_to_self_reports):
        #time.sleep(5)
        users_dict[d] = __get_dict(os.path.join(path_to_self_reports, d))

    with open('users_dict.pickle', 'wb') as handle:
        pickle.dump(users_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return  users_dict

def read_dict_from_pickle():
    with open('users_dict.pickle', 'rb') as handle:
        return pickle.load(handle)



