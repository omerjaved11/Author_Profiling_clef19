#!/usr/bin/python
import argparse
import pandas as pd
import emoji
import xml.etree.ElementTree as ET

import glob

import re
from emoji import UNICODE_EMOJI
import pickle
import os


def iter_docs(author):
    author_attr = author.attrib
    doc_dict = author_attr.copy()
    #    print(doc_dict)
    doc_dict['text'] = [' '.join([doc.text for doc in author.iter('document')])]

    return doc_dict


def create_data_frame(input_folder):
    os.chdir(input_folder)
    all_xml_files = glob.glob("*.xml")

    temp_list_of_DataFrames = []
    text_Data = pd.DataFrame()
    for file in all_xml_files:
        etree = ET.parse(file)  # create an ElementTree object
        doc_df = pd.DataFrame(iter_docs(etree.getroot()))
        doc_df['author_id'] = file[:-4]
        temp_list_of_DataFrames.append(doc_df)
    text_Data = pd.concat(temp_list_of_DataFrames, axis=0)

    return text_Data


def count_emoji(text):
    return len([c for c in text if c in UNICODE_EMOJI])


def face_smiling(text):
    return len([c for c in text if c in '😀😃😄😁😆😅🤣😂🙂🙃😉😊😇'])


def face_affection(text):
    return len([c for c in text if c in '🥰😍🤩😘😗☺😚😙'])


def face_tongue(text):
    return len([c for c in text if c in '😋😛😜🤪😝🤑'])


def face_hand(text):
    return len([c for c in text if c in '🤗🤭🤫🤔'])


def face_neutral_skeptical(text):
    return len([c for c in text if c in '🤐🤨😐😑😶😏😒🙄😬🤥'])


def face_concerned(text):
    return len([c for c in text if c in '😕😟🙁☹😮😯😲😳🥺😦😧😨😰😥😢😭😱😖😣😞'])


def monkey_face(text):
    return len([c for c in text if c in '🙈🙉🙊'])


def emotions(text):
    return len([c for c in text if c in '💋💌💘💝💖💗💓💞💕💟❣💔❤🧡💛💚💙💜🤎🖤'])


def preprocess(data):
    print('Preprocessing the Data')

    data['face_smiling'] = data['text'].apply(face_smiling)
    data['face_affection'] = data['text'].apply(face_affection)
    data['face_tongue'] = data['text'].apply(face_tongue)
    data['face_hand'] = data['text'].apply(face_hand)
    data['face_neutral_skeptical'] = data['text'].apply(face_neutral_skeptical)
    data['face_concerned'] = data['text'].apply(face_concerned)
    data['monkey_face'] = data['text'].apply(monkey_face)
    data['emotions'] = data['text'].apply(emotions)

    data['emoji_count'] = data['text'].apply(count_emoji)

    data['url_count'] = data['text'].apply(lambda x: len(re.findall('http\S+', x)))

    data['space_count'] = data['text'].apply(lambda x: len(re.findall(' ', x)))

    data['line_count'] = data['text'].apply(lambda x: len(re.findall('\n', x)))

    data['word_count'] = data['text'].apply(lambda x: len(re.findall('[a-zA-Z]', x)))

    data['capital_count'] = data['text'].apply(lambda x: len(re.findall('[A-Z]', x)))

    data['digits_count'] = data['text'].apply(lambda x: len(re.findall('[0-9]', x)))

    data['text_length'] = data['text'].apply(len)

    data['curly_brackets_count'] = data['text'].apply(lambda x: len(re.findall('[\{\}]', x)))

    data['round_brackets_count'] = data['text'].apply(lambda x: len(re.findall('[\(\)]', x)))

    data['round_brackets_count'] = data['text'].apply(lambda x: len(re.findall('\[\]', x)))

    data['underscore_count'] = data['text'].apply(lambda x: len(re.findall('[_]', x)))

    data['question_mark_count'] = data['text'].apply(lambda x: len(re.findall('[?]', x)))

    data['exclamation_mark_count'] = data['text'].apply(lambda x: len(re.findall('[!]', x)))

    data['dollar_mark_count'] = data['text'].apply(lambda x: len(re.findall('[$]', x)))

    data['ampersand_mark_count'] = data['text'].apply(lambda x: len(re.findall('[&]', x)))

    data['hash_count'] = data['text'].apply(lambda x: len(re.findall('[#]', x)))

    data['tag_count'] = data['text'].apply(lambda x: len(re.findall('[@]', x)))

    data['slashes_count'] = data['text'].apply(lambda x: len(re.findall('[/,\\\\]', x)))

    data['operator_count'] = data['text'].apply(lambda x: len(re.findall('[+=\-*%<>^|]', x)))

    data['punc_count'] = data['text'].apply(lambda x: len(re.findall('[\'\",.:;`]', x)))


def getArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input Directory Path", required=True)
    parser.add_argument("-o", "--output", help="Ouput Directory Path", required=True)
    args = parser.parse_args()

    print("input {} output {} ".format(
        args.input,
        args.output,
    ))

    return args.input, args.output




def writefiles(data, output, lang):
    # try:
    #     os.mkdir('output')
    # except Exception as e:
    #     print(e)
    try:
        os.chdir(output)
    except Exception as e:
        print(e)
    try:
        os.mkdir(lang)
    except Exception as e:
        print(e)
    try:
        os.chdir(lang)
    except Exception as e:
        print(e)

    for index, row in data.iterrows():
        print(row['author_id'], row['lang'], row['author'], row['gender'])
        root = ET.Element("author", id=row['author_id'], lang=row['lang'], type=row['author'], gender=row['gender'])
        tree = ET.ElementTree(root)
        tree.write(row['author_id'] + ".xml")

def runWithLang(input_folder,output_folder,lang):
    input_folder = input_folder+lang
    data = create_data_frame(input_folder)

    preprocess(data)

    if data.isnull().values.any():
        data.isnull().values.any()
        data.fillna(0, inplace=True)
    try:
        os.chdir(root)
    except Exception as e:
        print(e)
    print(os.getcwd())
    authormodel = pickle.load(open('models/'+lang+'/modelBotHuman', 'rb'))
    gendermodel = pickle.load(open('models/'+lang+'/modelMaleFemale', 'rb'))

    author = authormodel.predict(data.drop(['lang', 'text', 'author_id'], axis=1))
    gender = gendermodel.predict(data.drop(['lang', 'text', 'author_id'], axis=1))
    data['author'] = author
    data['gender'] = gender
    data.loc[data.author =='bot', 'gender'] ='bot'
    writefiles(data,output_folder,lang)




def main():
    global root

    input_folder,output_folder = getArg()
    # input_folder, output_folder = '/home/omer/oj/pan Research/test/', '/home/omer/oj/pan Research/output/'

    runWithLang(input_folder,output_folder,'en')
    runWithLang(input_folder,output_folder,'es')


if __name__ == "__main__":
    main()



