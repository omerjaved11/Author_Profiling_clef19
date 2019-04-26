# Pan Author Identification (Bots and Gender Profiling)

Identify Author of text on bases of their stylometry and writing style.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirments.txt
```

## Usage
#### To train model

python train.py -i 'trainingdatapath'
```python
python train.py -i '/input/train/data/'
```

#### To test model 

python test.py -i 'testdatapath' -o 'outputpath'

```python
python test.py -i '/input/test/data/'  -o '/output/'
```

## Features Selected :
```
1. emoji_count -> Count all kind Kind of emojis
2. face_smiling -> Count 😀😃😄😁😆😅🤣😂🙂🙃😉😊😇
3. face_affection -> Count 🥰😍🤩😘😗☺😚😙
4. face_tongue -> Count 😋😛😜🤪😝🤑
5. face_hand -> Count 🤗🤭🤫🤔
6. face_neutral_skeptical -> Count 🤐🤨😐😑😶😏😒🙄😬🤥
7. face_concerned -> Count 😕😟🙁☹😮😯😲😳🥺😦😧😨😰😥😢😭😱😖😣😞
8. monkey_face -> Count 🙈🙉🙊
9. emotions -> Count 💋💌💘💝💖💗💓💞💕💟❣💔❤🧡💛💚💙💜🤎🖤'
10. url_count -> Count all kind of link/urls
11. space_count -> Spaces count
12. capital_count -> Capital letter count
13. text_length -> Total length of messge
14. curly_brackets_count -> Count { }
15. round_brackets_count -> Count ( )
16. underscore_count -> Count _
17. question_mark_count -> Count ?
18. exclamation_mark_count -> Count !
19. dollar_mark_count -> Count $
20. ampersand_mark_count -> Count &
21. hash_count -> Count #
22. tag_count -> Count @
23. slashes_count -> Count Slashes // / \
24. operator_count -> Count Operators +-*/%<>^|
25. punc_count -> Count Puntuations '",.:;`
26. line_count -> Count nextlines \n
27. word_count -> Count Words A-Za-z
```


# Results for English:
______________________________________

### Predict Bot / Human

Classifier | Accuracy
---------------------|-------------------
'LogisticRegression' | 0.9158576051779935
 'RandomForestClassifier'| 0.9757281553398058
 'LinearSVC'| 0.8770226537216829
 'BernoulliNB'| 0.9239482200647249
 'MultinomialNB'| 0.8236245954692557
 'SVC'| 0.5056634304207119

Best Model **RandomForestClassifier**

 Author       | precision  |  recall  |  f1-score |  support
--------------|------------|----------|-----------|---------
         bot  |     0.98   |   0.97   |     0.98  |    622
       human  |     0.97   |   0.98   |    0.98   |    614
   micro avg  |     0.98   |   0.98   |   0.98    |   1236
   macro avg  |     0.98   |   0.98   |   0.98    |   1236
weighted avg  |     0.98   |   0.98   |   0.98    |   1236

__________________________________________________

### Predict Male / Female

Classifier | Accuracy
---------------------|-------------
'LogisticRegression'| 0.7265372168284789
 'RandomForestClassifier'| 0.8106796116504854
 'LinearSVC'| 0.6019417475728155
 'BernoulliNB'| 0.616504854368932
 'MultinomialNB'| 0.616504854368932
 'SVC'| 0.4967637540453074

Best Model **RandomForestClassifier**

Gender         | precision |   recall | f1-score  | support
---------------|---------- | -------- |-----------|-------
      female   |    0.79   |    0.85  |    0.82   |    311
        male   |    0.83   |    0.77  |    0.80   |    307
   micro avg   |    0.81   |    0.81  |    0.81   |    618
   macro avg   |    0.81   |    0.81  |    0.81   |    618
weighted avg   |    0.81   |    0.81  |    0.81   |    618

________________________________________

# Results for Spanish:

________________________________________

### Predict Bot / Human

Classifier | Accuracy
----------------|--------------------
'LogisticRegression'| 0.8433333333333334
 'RandomForestClassifier'| 0.9288888888888889
 'LinearSVC'| 0.7488888888888889
 'BernoulliNB'| 0.8188888888888889
 'MultinomialNB'| 0.7644444444444445
 'SVC'| 0.4888888888888889

Best Model **RandomForestClassifier**

Author       |  precision  |  recall |  f1-score|   support
------       |-------------|---------|----------|----------
         bot |       0.93  |    0.93 |     0.93 |      440 
       human |      0.93   |   0.93  |    0.93  |     460  
   micro avg |      0.93   |   0.93  |    0.93  |     900  
   macro avg |      0.93   |   0.93  |    0.93  |     900  
weighted avg |      0.93   |   0.93  |    0.93  |     900  

________________________________________

### Predict Male / Female

Classifier | Accuracy
------------|---------------------
'LogisticRegression'| 0.6844444444444444
 'RandomForestClassifier' | 0.7844444444444445
 'LinearSVC'| 0.5666666666666667
 'BernoulliNB'| 0.6066666666666667
 'MultinomialNB'| 0.6355555555555555
 'SVC'| 0.48444444444444446

Best Model **RandomForestClassifier**


Gender         | precision |   recall |  f1-score |  support
-------------- | --------- | -------- | --------- | ------
      female   |    0.77   |   0.83   |   0.80    |   232
        male   |    0.80   |   0.74   |   0.77    |   218
   micro avg   |    0.78   |   0.78   |   0.78    |   450
   macro avg   |    0.79   |   0.78   |   0.78    |   450
weighted avg   |    0.79   |   0.78   |   0.78    |   450



_________________________________

