from nltk.corpus import stopwords
import glob
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
stops = stopwords.words('english')
punct = ["'", ".", ",", '"', ";", "?", "!", "-", "-", "(", ")", "*", "/", ":", "~"]

# Store all content words in each review
stars = {"1_star": [], "2_star": [], "3_star": [], "4_star": [], "5_star": [],}

# Store the values obtained from Textblob from training data
sentiment_vals = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}

def get_data():
  alltext = "" 
  toksents = []  
  f = open("../json_data/tokenize/first_10k/all_reviews.json")
  data = json.loads(f.read().rstrip())
  f.close()
  allsent = []
  for review in data:
    sent = review["text"]
    allsent = sent_tokenize(sent)
    for w in allsent:
      toksents.append(nltk.word_tokenize(w))
  return(toksents)
get_data()
model = Word2Vec(get_data(), size=100, window=5, min_count=3, workers=4)
print("model built!")
model.wv['first']