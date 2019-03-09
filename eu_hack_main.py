import io
import os
import json

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account

def get_dict():
    path_to_json = os.path.join(os.path.dirname(PATH),'NY Taxis-a3cc565a66b6.json')
    path_to_self_reports = os.path.join(os.path.dirname('dataset', 'bigo', 'self_reports'))
    credentials = service_account.Credentials. from_service_account_file(path_to_json)

    client = vision.ImageAnnotatorClient(credentials=credentials)

    user_dict={}
    for subdir, dirs, files in os.walk(path_to_self_reports):
        user_dict[subdir]={}
        for file in files:
            if file.endswith('jpg'):
                with io.open(file, 'rb') as image_file:
                    content = image_file.read()
                    image = types.Image(content=content)
                    # Performs label detection on the image file
                    response = client.label_detection(image=image)
                    labels = response.label_annotations
                    user_dict[subdir][file[-3:]=labels


    for user_key, value in user_dict.items():
        print(user_key)
        for meal_key, meal_labels in value.items():
            for label in meal_labels:
                print(label)