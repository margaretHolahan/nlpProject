## Restaurant Review Star Prediction for Yelp Dataset 
In this repository, we provide sentiment analysis on yelp restaurant reviews to predict star ratings(1-5 stars). We investigate the data set provided by Yelp Dataset. Eight machine learning models are used. Statistical classifiers: Naive Bayes, Smoothing Naive Bayes, Gaussian Naive Bayes, Logistic Regression, KNN. Neural network models: FFNN, LSTM. Transformer: DistilBERT. After analyzing the performance of each model, the best model for predicting the ratings from reviews is DistilBERT.  

## Dataset
7M Total Reviews from Yelp Dataset \
3.6M Restaurant Reviews Found\
5.6M Total Word Count  \
78k Total Sentence Count


|   | stars  | text |
| :------------ |:---------------:| -----:|
| 0  | 1| We have always enjoyed this restaurant and the... |
| 1  | 2      | Not terrible, just not good. Steak was pretty...|
| 2 |  4  |   I've often passed this swanky little place in ... |
| 3 | 1|   I just got back from Chucks and had to login t... |
| 4 | 5 |   There are few truly great lunch places in Old ... |

### F1 Results Not Shown on Poster

| Stars | Gaussian NB F1 | Logistic Regression F1 |
| :------------ |:---------------:| :-----:|
| 1 | 0.25 | 0.07 
| 2 | 0.14 | 0.20
| 3 | 0.23 | 0.09
| 4 | 0.14 | 0.31
| 5 | 0.38 | 0.16


## Classifiers
All the classifier programs are contained in the folder

**Classifiers using Bag of Words Features:**\
Naive Bayes, Smoothing Naive Bayes, LSTM

**Classifiers using Word Embeddings:**\
Gaussian Naive Bayes, Logistic Regression, KNN, FFNN

## Filter Programs
Contains all python scripts we used to filter the Yelp Open Dataset (https://www.yelp.com/dataset)
Also includes python scripts used to preprocess our data 

## JSON Data
Contains all filtered JSON files (filtered with scripts in filter programs) used for our classifiers

