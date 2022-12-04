import glob
import json
import selectors
from nltk.corpus import stopwords
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
stops = stopwords.words('english')
punct = ["'", ".", ",", '"', ";", "?", "!", "-", "-", "(", ")", "*", "/", ":", "~"]

def get_data():
    # Files storing preprocessed testing and training data for 1-5 star reviews
    # Preprocessing = make all text lower case and wrap punctuation marks with a space
    all_files = glob.glob("../json_data/tokenize/first_10k/all_reviews")
    
    for file in all_files:
        star_fd = open(file, 'r', encoding='utf-8')
        data = json.loads(star_fd.read())

        features = []
        target = []

        for review in data:
            thesewords = []
            words = review["text"].rstrip().split()
            stars = review["stars"]
            for w in words:
                for s in stars:
                    if w not in stops and w not in punct and w not in thesewords:
                     features.append(w)
                     target.append(s)
                     thesewords.append(w)
            star_fd.close()
            return [features, target]
def knn():
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    X=get_data()[0]
    Y=get_data()[1]
    gnb = GaussianNB()
    scores = cross_validate(gnb, X, Y, cv=10, scoring=scoring_metrics)
    for score_name, score_value in scores.items():
        print(score_name, score_value)

knn()