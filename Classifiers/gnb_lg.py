# Program to calculate Gaussian NB and Logistic Regression with
# Word2Vec Word Embeddings
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import numpy as np
import nltk
import glob
import json

# Stopwords 
stops = stopwords.words('english')
punct = ["'", ".", ",", '"', ";", "?", "!", "-", "-", "(", ")", "*", "/", ":", "~", "``"]

# Files 
train_files = glob.glob("./json_data/tokenize/first_10k/train/*")
test_files = glob.glob("./json_data/tokenize/first_10k/train/*")

def get_from_data(files, list_wanted):
  tokenized = []  
  target_stars = []

  for file in files:
    fd = open(file, 'r', encoding='utf-8')
    data = json.loads(fd.read())
    for review in data:
      all_sents = review["text"]
      tok_sents = sent_tokenize(all_sents)
      cur_review = []
      for sent in tok_sents:
        words = nltk.word_tokenize(sent)
        for w in words:
          if w not in stops and w not in punct:
            cur_review.append(w)
      tokenized.append(cur_review)
      target_stars.append(review['stars'])
    fd.close()

    # print(len(tokenized) == len(target_stars))

  if list_wanted == "tok":
    return tokenized

  return target_stars

def make_vectors(filenames):
  vectors = []
  data = get_from_data(filenames, "tok")
  model = Word2Vec(data, window=5, min_count=1, workers=4)

  # Create a one vector for each review    
  for review in data:
    totvec = np.zeros(100)
    for word in review:
      totvec = totvec + model.wv[word]
    vectors.append(totvec)
  return vectors

def run_classifiers():
  # Get Vectors and Real Stars  
  trained_files_v = make_vectors(train_files)
  trained_stars = get_from_data(train_files, "stars")

  test_files_v = make_vectors(test_files)
  target_stars = get_from_data(test_files, "stars")

  # Gaussian NB
  nb_classifier = GaussianNB()
  nb_classifier.fit(trained_files_v, trained_stars)
  nb_pred = nb_classifier.predict(test_files_v)

  # Logistic Regression
  logistic_regression = LogisticRegression(C=25.0,solver='lbfgs',multi_class='auto',max_iter=10000)
  logistic_regression.fit(trained_files_v, trained_stars)
  lg_pred = logistic_regression.predict(test_files_v)

  # Print Results
  # Gaussian NB
  print("Gaussian NB")
  
  print(classification_report(target_stars, nb_pred))
  print(confusion_matrix(target_stars, nb_pred))

  # Logistic Regression
  print("Logistic Regression")
  print(classification_report(target_stars, lg_pred))
  print(confusion_matrix(target_stars, lg_pred))

run_classifiers()