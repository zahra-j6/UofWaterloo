import pandas as pd
from pandas import DataFrame, read_csv
import os # std python3 lib
import re #std python3 lib
import csv
import numpy as np
import nltk
from nltk.corpus import stopwords

inpath = "data/aclImdb/train/"  # source data
test_path = "data/aclImdb/test/"  # test data for grade evaluation.

# using the concept of stopwords which is meant to remove most commonly occuring words to improve performance
def remove_stopwords(sentence):
    sentencewords = sentence.split()
    stopwords=set(nltk.corpus.stopwords.words("english"))
    resultwords = [word for word in sentencewords if not word  in stopwords]
    result = ' '.join(resultwords)
    return result

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
   # re.sub('<[^>]+>','',text)
    return TAG_RE.sub('', text)

def preprocessing(data):
#    stopwords = open('stopwords.en.txt', 'r', encoding="ISO-8859-1").read()
#    stopwords = stopwords.split("\n")
    data = remove_stopwords(data) # remove stopwords
    data = remove_tags(data)  # remove html tags
    data=re.sub('[^a-zA-Z]', ' ', data)#Remove punctuations and numbers
    data=re.sub(r'[^\w\s]','',data, re.UNICODE) #remove alphanumeric
    data=data.lower()# converting data to lower case
    data=re.sub('\s{2,}',' ',data) # remove multiple spaces
    return data

def preparing(name,mix=False):
    if name=='imdb_train3.csv':

        indices = []
        text = []
        rating = []
        sentiment=[]

        i = 0

        for filename in os.listdir(inpath + "pos"):
            data = open(inpath + "pos/" + filename, 'r', encoding="ISO-8859-1").read()
            data=preprocessing(data)
            indices.append(i)
            text.append(data)
            rating.append("1")
            sentiment.append('positive')
            i = i + 1

        for filename in os.listdir(inpath + "neg"):
            data = open(inpath + "neg/" + filename, 'r', encoding="ISO-8859-1").read()
            data=preprocessing(data)
            indices.append(i)
            text.append(data)
            rating.append("0")
            sentiment.append('negative')
            i = i + 1

        Dataset = list(zip(indices, text, rating,sentiment))

        if mix:
            np.random.shuffle(Dataset)

        df = pd.DataFrame(data=Dataset, columns=['row_Number', 'text', 'rating','sentiment'])
        df.to_csv("./" + name, index=False, header=True)
    elif name=='imdb_test3.csv':

        indices = []
        text = []
        rating = []
        sentiment=[]

        i = 0

        for filename in os.listdir(test_path + "pos"):
            data = open(test_path + "pos/" + filename, 'r', encoding="ISO-8859-1").read()
            data=preprocessing(data)
            indices.append(i)
            text.append(data)
            rating.append("1")
            sentiment.append('positive')
            i = i + 1

        for filename in os.listdir(test_path + "neg"):
            data = open(test_path + "neg/" + filename, 'r', encoding="ISO-8859-1").read()
            data=preprocessing(data)
            indices.append(i)
            text.append(data)
            rating.append("0")
            sentiment.append('negative')
            i = i + 1

        Dataset = list(zip(indices, text, rating,sentiment))

        if mix:
            np.random.shuffle(Dataset)

        df = pd.DataFrame(data=Dataset, columns=['row_Number', 'text', 'rating','sentiment'])
        df.to_csv("./" + name, index=False, header=True)


