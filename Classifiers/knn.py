from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd 
from sklearn import metrics
from nltk.corpus import stopwords
import glob
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from get_data.py import get_vectors, get_data

def knn():
  knn = KNeighborsClassifier()
  #call it something other than model
  knn.fit(get_vectors("train"), get_data("train", False))
  expected = get_data("test", False)
  predicted = knn.predict(get_vectors("test"))
  print(metrics.classification_report(expected, predicted))
  print(metrics.confusion_matrix(expected, predicted))
knn()
def svc():
  svc = 