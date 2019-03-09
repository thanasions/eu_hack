import eu_hack_as as eu
import tags as tg


def tag_them(users_dict):
    tagger = tg.Tags()
    for key, values in users_dict.items():
        print('user: ' + key)
        for idx, labels in values.items():
            print(str(idx) + ": " + str(labels))
            ret_tags = tagger.calculate(labels)
            print(ret_tags)


def main():
    users_dict = eu.read_dict_from_pickle()
    tags2 = tag_them(users_dict)


if __name__ == '__main__':
    main()
