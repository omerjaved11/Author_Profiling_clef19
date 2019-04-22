#!/usr/bin/python
import os
import argparse
import pandas as pd
import emoji
import xml.etree.ElementTree as ET
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def iter_docs(author):
    author_attr = author.attrib
    doc_dict = author_attr.copy()
#    print(doc_dict)
    doc_dict['text'] = [' '.join([doc.text for doc in author.iter('document')])]
        
    return doc_dict


def create_data_frame(input_folder):

	os.chdir(input_folder)
	all_xml_files=glob.glob("*.xml")
	truth_data = pd.read_csv('truth.txt',sep=':::',names=['author_id','author','gender'])

	temp_list_of_DataFrames = []
	text_Data = pd.DataFrame()
	for file in all_xml_files:
		 etree = ET.parse(file) #create an ElementTree object
		 doc_df =pd.DataFrame(iter_docs(etree.getroot()))
		 doc_df['author_id'] = file[:-4]
		 temp_list_of_DataFrames.append(doc_df)
	text_Data = pd.concat(temp_list_of_DataFrames, axis=0)

	data = text_Data.merge(truth_data,on='author_id')
	return data


import re
from emoji import UNICODE_EMOJI

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
    return len([c for c in text if c in  '💋💌💘💝💖💗💓💞💕💟❣💔❤🧡💛💚💙💜🤎🖤'])



def preprocess(data):
	#     data['emoji_count'] = data['text'].apply(count_emoji)
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

	return args.input,args.output
# dot, single qoute, double qoute, comma


def main():



	root = os.getcwd()

	input_folder,output_folder = getArg()

	data = create_data_frame(input_folder)
	# print(data)



	preprocess(data)

	if data.isnull().values.any():
		data.isnull().values.any()
		data.fillna(0, inplace=True)


	X_train, X_test, y_train, y_test = train_test_split(data.drop(['lang','text','author_id','gender','author'],axis=1),data['author'],test_size=0.3)

	from sklearn.ensemble import RandomForestClassifier
	RandomForestClassifier = RandomForestClassifier()
	RandomForestClassifier.fit(X_train, y_train)
	RandomForestClassifier_prediction = RandomForestClassifier.predict(X_test)
	accuracy_RandomForestClassifier = accuracy_score(y_test, RandomForestClassifier_prediction)

	print('Accuracy: ',accuracy_RandomForestClassifier)

	print(classification_report(y_test,RandomForestClassifier_prediction))

if __name__ == "__main__":
	main()



