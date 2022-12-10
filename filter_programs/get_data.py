
from nltk.corpus import stopwords
import glob
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/nlpProject/all_reviews.csv')
shuffled_df = df.sample(frac =1)
shuffled_df.to_csv("shuffled_data.csv", index=False)

stops = stopwords.words('english')
punct = ["'", ".", ",", '"', ";", "?", "!", "-", "-", "(", ")", "*", "/", ":", "~"]

def get_data(name, answer):
  alltext = "" 
  toksents = []  
  stars = []
  all_files = glob.glob("../json_data/tokenize/first_10k/" + name + "/*")
  for file in all_files:
    f = open(file)
    data = json.loads(f.read().rstrip())
    f.close()
    allsent = []
    for review in data:
      sent = review["text"]
      allsent = sent_tokenize(sent)
      for w in allsent:
        if(w not in stops):
          toksents.append(nltk.word_tokenize(w))
          stars.append(review['stars'])
  if(answer == True):
    return(toksents)
  else:
    return(stars)
def get_all_data(answer):
  alltext = "" 
  toksents = []  
  stars = []
  f = open("../json_data/tokenize/first_10k/all_reviews.json")
  data = json.loads(f.read().rstrip())
  f.close()
  allsent = []
  for review in data:
    sent = review["text"]
    allsent = sent_tokenize(sent)
    for w in allsent:
      if(w not in stops):
        toksents.append(nltk.word_tokenize(w))
        stars.append(review['stars'])
  if(answer == True):
    return(toksents)
  else:
    return(stars)
def get_vectors(name):
  vectors = []
  data = get_data(name, True)
  model = Word2Vec(data, window=5, min_count=3, workers=4)
  for d in data:
    totvec = np.zeros(100)
    for w in d:
      if w.lower() in model:
        totvec = totvec + model[w.lower()]
    vectors.append(totvec)
  return vectors