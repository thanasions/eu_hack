import io
import os
import json

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
#/ dataset/ bigO/ self_reports/ 820/ daily_answers/ 0.json
def get_dict():
    path_to_json = 'NY Taxis-a3cc565a66b6.json'
    path_to_self_reports = os.path.join('dataset', 'big0', 'self_reports')
    credentials = service_account.Credentials. from_service_account_file(path_to_json)

    client = vision.ImageAnnotatorClient(credentials=credentials)

    user_dict={}

    for user_dir in os.listdir(path_to_self_reports):
        user_dict[user_dir]={}
        dirs = os.listdir(os.path.join(path_to_self_reports,user_dir))
        if 'meals' in dirs:
            dirs_dirs = os.listdir(os.path.join(dirs, user_dir, 'meals'))
            for file in dirs_dirs:
                if file.endswith('.jpg'):
                    with io.open(file, 'rb') as image_file:
                        content = image_file.read()
                        image = types.Image(content=content)
                        # Performs label detection on the image file
                        response = client.label_detection(image=image)
                        labels = response.label_annotations
                        meal_number = file.strplit(".")[0]
                        user_dict[user_dir][meal_number] = [ l.description for l in labels ]
