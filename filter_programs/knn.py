import glob
import json
import selectors
from nltk.corpus import stopwords
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
stops = stopwords.words('english')

punct = ["'", ".", ",", '"', ";", "?", "!", "-", "-", "(", ")", "*", "/", ":", "~"]

def get_data(name):
    # Files storing preprocessed testing and training data for 1-5 star reviews
    # Preprocessing = make all text lower case and wrap punctuation marks with a space
     all_files = glob.glob("../json_data/first_10k/all_reviews.json")
    
     for file in all_files:
        star_fd = open(file, 'r', encoding='utf-8')
        data = json.loads(star_fd.read())

        features = []
        target = []
        adding = []
        for review in data:
            thesewords = []
            words = review["text"].rstrip().split()
            target.append(review["stars"])
            for w in words:
                  if w not in stops and w not in punct and w not in thesewords:
                    adding.append(w)
                    thesewords.append(w)
            features.append(adding)
            star_fd.close()
        if name == "features":
          return features
        if name == "target":
          return target
def knn():
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    X=get_data("features")
    Y=get_data("target")
    MinMaxScaler = preprocessing.MinMaxScaler()
    X_data_minmax = MinMaxScaler.fit_transform(X)
    data = pd.DataFrame(X_data_minmax,columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state = 1)
    knn_clf=KNeighborsClassifier()
    knn_clf.fit(X_train,y_train)
    ypred=knn_clf.predict(X_test) 
knn()