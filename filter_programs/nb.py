# Program to read find NB probability for 1-5 star reviews 
# Based off nb.py file from Pset 1 
from os import curdir
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from threading import current_thread
from nltk import FreqDist
from nltk.corpus import stopwords
import glob
import math
import re
from textblob import TextBlob
import json
import random
import numpy as np

# Stopwords
stops = stopwords.words('english')
punct = ["'", ".", ",", '"', ";", "?", "!", "-", "-", "(", ")", "*", "/", ":", "~"]

# Store all content words in each review
stars = {"1_star": [], "2_star": [], "3_star": [], "4_star": [], "5_star": [],}

# Store the values obtained from Textblob from training data
sentiment_vals = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}

def get_training_data():
    # Files storing preprocessed testing and training data for 1-5 star reviews
    # Preprocessing = make all text lower case and wrap punctuation marks with a space
    all_files = glob.glob("./json_data/tokenize/first_10k/train/*")
    
    for file in all_files:
        star_fd = open(file, 'r', encoding='utf-8')
        data = json.loads(star_fd.read())

        # We want to find all unique words within each review (NOT all reviews)
        # Use these words to check repetitions and append to allwords
        allwords = []

        for review in data:
            thesewords = []
            words = review["text"].rstrip().split()
            for w in words:
                if w not in stops and w not in punct and w not in thesewords:
                    allwords.append(w)
                    thesewords.append(w)
        star_fd.close()

        # Find which review we are working with 1 star, 2 star etc.
        for key in stars:
            if key in file:
                this_star = key
                # If key found, break out of loop
                continue
        stars[this_star] = allwords

# Find what the sentiment range is for each star file and store 
def find_sentiment_range():
    for star in stars:
        cur_review = TextBlob(' '.join(stars[star]))
        cur_sentiment = cur_review.sentiment.polarity
        sentiment_vals[star] = cur_sentiment

def calculate_nb_probabilities():
    all_freqs = {
        "1_star" : FreqDist(stars["1_star"]),
        "2_star" : FreqDist(stars["2_star"]),
        "3_star" : FreqDist(stars["3_star"]),
        "4_star" : FreqDist(stars["4_star"]),
        "5_star" : FreqDist(stars["5_star"])
    }

    all_probs = {
        "1_star" : {},
        "2_star" : {},
        "3_star" : {},
        "4_star" : {},
        "5_star" : {}
    }

    # For each freq dist from each star
    for star in all_freqs:
        cur_freq = all_freqs[star]
        total_count = len(stars[star])
        # Find nb probabilities
        for word in cur_freq:
            word_count = cur_freq[word]
            cur_prob = math.log(word_count/total_count)
            all_probs[star][word] = cur_prob

    return all_probs

def naive_bayes(reviewwords, all_probs):
    defaultprob = math.log(0.0000000000001)
    
    test_scores = {
        "1_star" : None,
        "2_star" : None,
        "3_star" : None,
        "4_star" : None,
        "5_star" : None
    }

    for star in all_probs:
        # print(reviewwords[0])
        test_scores[star] = all_probs[star].get(reviewwords[0], defaultprob)
        # print(all_probs[star].get(reviewwords[0]))
        # print(test_scores[star])
        for i in range(1, len(reviewwords)):
            # Multiply probabilities of each word to get total nb for the review
            test_scores[star] *= all_probs[star].get(reviewwords[i], defaultprob)

    # Find star with highest probability
    max_star = None
    max = None
    for star in test_scores:
        if max == None:
            max = test_scores[star]
            max_star = star
        if max > test_scores[star]:
            max = test_scores[star]
            max_star = star

    return max_star


def calculate_smooth_nb_probabilities():
    # Smoothed content word probabilities 
    smoothed_probs = {"1_star" : {}, "2_star" : {}, "3_star" : {}, "4_star" : {}, "5_star" : {}}
    total_types = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}
    total_tokens = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}
    
    
    smoothed_freqs = {
        "1_star" : FreqDist(stars["1_star"]),
        "2_star" : FreqDist(stars["2_star"]),
        "3_star" : FreqDist(stars["3_star"]),
        "4_star" : FreqDist(stars["4_star"]),
        "5_star" : FreqDist(stars["5_star"])
    }

    for star in stars:
        total_types[star] = len(set(stars[star]))
        total_tokens[star] = len(stars[star])
    
    for star in smoothed_freqs:
        for word in smoothed_freqs[star]:
            word_count = smoothed_freqs[star][word] + 1
            cur_prob = math.log(word_count/(total_tokens[star] + total_types[star] + 1))
            smoothed_probs[star][word] = cur_prob

    return smoothed_probs, total_types, total_tokens


def smooth_naive_bayes(reviewwords):
    smoothed_probs, total_types, total_tokens = calculate_smooth_nb_probabilities()
    default_prob = {"1_star": 0, "2_star": 0, "3_star": 0, "4_star": 0, "5_star": 0}
    scores = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}
    
    for star in stars:
        default_prob[star] = math.log(1 / (total_types[star] + total_tokens[star] + 1))

    for star in stars:
        scores[star] = smoothed_probs[star].get(reviewwords[0], default_prob[star])
        for i in range(1, len(reviewwords)):
            scores[star] *= smoothed_probs[star].get(reviewwords[i], default_prob[star])

    # Find star with highest probability
    max_star = None
    max = None
    for star in scores:
        if max == None:
            max = scores[star]
            max_star = star
        if max > scores[star]:
            max = scores[star]
            max_star = star

    return max_star


def calculate_accuracy(all_probs):
    true_results = {"nb": []}
    pred_results = {"nb": []}
    nbcorrect = 0
    smnbcorrect = 0
    tbcorrect = 0
    affcorrect = 0

    star_list = ["1_star", "2_star", "3_star", "4_star", "5_star"]
    all_files = glob.glob("./json_data/tokenize/first_10k/test/*")
    for file in all_files:
        star_fd = open(file, 'r', encoding='utf-8')
        data = json.loads(star_fd.read())

        for review in data:
            testwords = []
            words = set(review["text"].rstrip().split())
            for w in words:
                if w not in stops and w not in punct:
                    testwords.append(w)
        
            # Find which review we are working with 1 star, 2 star etc.
            for star in star_list:
                if star in file:
                    real_star = star
                    # If star found, break out of loop
                    continue
            
            # Apply each classifier to review and check to see if correct
            nbstar = naive_bayes(testwords, all_probs)
            if real_star == nbstar:
                nbcorrect += 1

            true_results["nb"].append(real_star)
            pred_results["nb"].append(nbstar)

            if real_star == smooth_naive_bayes(testwords):
                smnbcorrect += 1
        star_fd.close()

    cm = confusion_matrix(true_results["nb"], pred_results["nb"], labels=["1_star", "2_star", "3_star", "4_star", "5_star"])
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)

    print("Naive Bayes Accuracy: ", (nbcorrect/1600))
    print("Naive Bayes Precision: ", np.mean(precision))
    print("Naive Bayes Recall: ", np.mean(recall))
    print("Smoothed Naive Bayes Accuracy: ", (smnbcorrect/1600))


# MAIN SECTION #
get_training_data()
find_sentiment_range()

all_probs = calculate_nb_probabilities()

# print(train_scores)
calculate_accuracy(all_probs)

print(sentiment_vals)