#!/usr/bin/python
import argparse
import pandas as pd
import emoji
import xml.etree.ElementTree as ET
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
    truth_data = pd.read_csv('truth.txt', sep=':::', names=['author_id', 'author', 'gender'])

    temp_list_of_DataFrames = []
    text_Data = pd.DataFrame()
    for file in all_xml_files:
        etree = ET.parse(file)  # create an ElementTree object
        doc_df = pd.DataFrame(iter_docs(etree.getroot()))
        doc_df['author_id'] = file[:-4]
        temp_list_of_DataFrames.append(doc_df)
    text_Data = pd.concat(temp_list_of_DataFrames, axis=0)

    data = text_Data.merge(truth_data, on='author_id')
    return data


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
    parser.add_argument("-o", "--output", help="Ouput Directory Path" )
    args = parser.parse_args()

    print("input {} output {} ".format(
        args.input,
        args.output,
    ))

    return args.input, args.output


# dot, single qoute, double qoute, comma

def getBestModel(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    LogisticRegression = LogisticRegression()
    from sklearn.ensemble import RandomForestClassifier
    RandomForestClassifier = RandomForestClassifier()
    from sklearn.svm import LinearSVC
    LinearSVC = LinearSVC()
    from sklearn.naive_bayes import BernoulliNB
    BernoulliNB = BernoulliNB()
    from sklearn.naive_bayes import MultinomialNB
    MultinomialNB = MultinomialNB()
    from sklearn.svm import SVC
    SVC = SVC(kernel='rbf')

    models1 = {'LogisticRegression': LogisticRegression,
               'RandomForestClassifier': RandomForestClassifier,
               'LinearSVC': LinearSVC,
               'BernoulliNB': BernoulliNB,
               'MultinomialNB': MultinomialNB,
               'SVC': SVC}

    predictions = {}
    accuracy = {}

    for model in models1:
        print(models1[model])
        models1[model].fit(X_train, y_train)
        predictions[model] = models1[model].predict(X_test)
        accuracy[model] = accuracy_score(y_test, predictions[model])

    print(accuracy)
    print('Best Model', max(accuracy, key=accuracy.get))
    print(classification_report(y_test, predictions[max(accuracy, key=accuracy.get)]))
    model = models1[max(accuracy, key=accuracy.get)]
    return model


def buildModels(model, features, classLabel, modelname,lang):
    model.fit(features, classLabel)
    print(root)
    try:
        os.chdir(root)
        print('Change current Dir to '+root)
    except Exception as e:
        print(e)

    try:

        os.mkdir('models')
        print('Make Dir to models')
    except Exception as e:
        print(e)

    try:
        os.chdir('models')
        print('Change current Dir to models')
    except Exception as e:
        print(e)

    try:
        os.mkdir(lang)
        print('Make Dir '+lang)

    except Exception as e:
        print(e)
    try:
        os.chdir(lang)
        print('Change current Dir to '+lang)

    except Exception as e:
        print(e)

    print('writing model')
    pickle.dump(model, open(modelname, 'wb'))

    try:
        os.chdir(root)
        print('Change current Dir to '+root)

    except Exception as e:
        print(e)


def runWithLang(input_folder,lang):
    input_folder = os.path.join(input_folder,lang)
    data = create_data_frame(input_folder)
    # print(data)9mOnelRy

    preprocess(data)

    if data.isnull().values.any():
        data.isnull().values.any()
        data.fillna(0, inplace=True)

#model for bot/human
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['lang', 'text', 'author_id', 'gender', 'author'], axis=1), data['author'], test_size=0.3)

    model = getBestModel(X_train, X_test, y_train, y_test)

    features = data.drop(['lang', 'text', 'author_id', 'gender', 'author'], axis=1)

    classLabel = data['author']
    print('Building Model for Bot vs Human')
    buildModels(model, features, classLabel, 'modelBotHuman',lang)

#model for male/human
    temp_data = data[data.gender != 'bot']
    print('bot' in temp_data.gender)

    X_train, X_test, y_train, y_test = train_test_split(
        temp_data.drop(['lang', 'text', 'author_id', 'gender', 'author'], axis=1), temp_data['gender'], test_size=0.3)

    model = getBestModel(X_train, X_test, y_train, y_test)

    features = temp_data.drop(['lang', 'text', 'author_id', 'gender', 'author'], axis=1)

    classLabel = temp_data['gender']
    print('Building model for Male vs Female')
    buildModels(model, features, classLabel, 'modelMaleFemale', lang)




def main():
    global root

    root = os.getcwd()

    input_folder,output_folder = getArg()
    # input_folder, output_folder = '/home/omer/Documents/pan stuff/data/', '/output'

    runWithLang(input_folder,'en')
    runWithLang(input_folder,'es')


if __name__ == "__main__":
    main()



