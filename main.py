import io
import os
import eu_hack_as as eu
import tags as tg
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def train(users_dict):
    tagger = tg.Tags(file_name='models/doc2vec.model', )
    data=[]
    for key, values in users_dict.items():
        print('user: ' + key)
        for idx, labels in values.items():
            data.append(' '.join(labels))
    tagger.traindoc2vec(data)


def tag_them(users_dict):

    PATH = os.getcwd()
    path_to_self_reports = os.path.join(PATH, 'dataset', 'big0', 'self_reports')
    tagger = tg.Tags()#file_name='models/fasttext/wiki-news-300d-1M-subword.vec'
    for key, values in users_dict.items():
        print('user: ' + key)
        for idx, labels in values.items():
            print(str(idx) + ": " + str(labels))
            ret_tags = tagger.calculate_word(labels)
            print(ret_tags)

            img_path = os.path.join(path_to_self_reports, key, 'meals', idx+'.jpg' )
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype("/opt/X11/share/fonts/TTF/VeraIt.ttf", 36)
            # draw.text((x, y),"Sample Text",(r,g,b))
            counter=1
            for ret_key, ret_val in ret_tags.items():
                draw.text(( 30, counter *50), str(ret_key) +": " + str(ret_val), (255, 255, 255), font=font)
                counter+=1
            font = ImageFont.truetype("/opt/X11/share/fonts/TTF/VeraIt.ttf", 26)
            for label in labels:
                draw.text((230, counter * 50),  str(label), (255, 255, 255), font=font)
                counter += 1

            img.save('imgs_glove/'+str(key)+"_" + str(idx) +'.jpg')


def main():
    users_dict = eu.read_dict_from_pickle()
    tags2 = tag_them(users_dict)


if __name__ == '__main__':
    main()
