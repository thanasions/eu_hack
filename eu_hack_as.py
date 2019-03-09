
# coding: utf-8

# In[1]:


import io
import os
import json

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account


# In[20]:


PATH = os.getcwd()
path_to_json = os.path.join(PATH,'NY Taxis-a3cc565a66b6.json')
path_to_self_reports = os.path.join(PATH, 'dataset', 'big0', 'self reports')


# In[21]:


path_to_self_reports


# In[9]:


credentials = service_account.Credentials.from_service_account_file(path_to_json)
client = vision.ImageAnnotatorClient(credentials=credentials)


# In[29]:


def get_dict(user_dir):
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


# In[30]:


os.listdir(path_to_self_reports)


# In[31]:


users_dict={}
for d in os.listdir(path_to_self_reports):
    users_dict[d] = get_dict(os.path.join(path_to_self_reports, d))


# In[32]:


users_dict


# In[ ]:




