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

    for user_dir in os.listdir(path_to_self_reports):
        user_dict[subdir]={}
        if 'meals' in os.listdir(os.path.join(path_to_self_reports, user_dir)):
            for file in os.listdir(os.path.join(path_to_self_reports, user_dir, 'meals')):
                if file.endswith('jpg'):
                    with io.open(file, 'rb') as image_file:
                        content = image_file.read()
                        image = types.Image(content=content)
                        # Performs label detection on the image file
                        response = client.label_detection(image=image)
                        labels = response.label_annotations
                        meal_number = file.strplit(".")[0]
                        user_dict[subdir][meal_number] = [ l.description for l in labels ]
