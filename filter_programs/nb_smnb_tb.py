# Program to create Naive Bayes, Smoothing Naive Bayes, and TextBlob classifiers
# To categorize test reviews into 1-5 star reviews 
# Based off nb.py file from Pset 1 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk import FreqDist
from textblob import TextBlob
import matplotlib.pyplot as plt
import json
import numpy as np
import glob
import math
import random

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
            # Multiply probabilities of each word to get total nb for the current review
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
    # Smoothing content word probabilities 
    smoothing_probs = {"1_star" : {}, "2_star" : {}, "3_star" : {}, "4_star" : {}, "5_star" : {}}
    total_types = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}
    total_tokens = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}
    
    smoothing_freqs = {
        "1_star" : FreqDist(stars["1_star"]),
        "2_star" : FreqDist(stars["2_star"]),
        "3_star" : FreqDist(stars["3_star"]),
        "4_star" : FreqDist(stars["4_star"]),
        "5_star" : FreqDist(stars["5_star"])
    }

    for star in stars:
        total_types[star] = len(set(stars[star]))
        total_tokens[star] = len(stars[star])
    
    for star in smoothing_freqs:
        for word in smoothing_freqs[star]:
            word_count = smoothing_freqs[star][word] + 1
            cur_prob = math.log(word_count/(total_tokens[star] + total_types[star] + 1))
            smoothing_probs[star][word] = cur_prob

    return smoothing_probs, total_types, total_tokens

def smooth_naive_bayes(reviewwords):
    smoothing_probs, total_types, total_tokens = calculate_smooth_nb_probabilities()
    default_prob = {"1_star": 0, "2_star": 0, "3_star": 0, "4_star": 0, "5_star": 0}
    scores = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}
    
    for star in stars:
        default_prob[star] = math.log(1 / (total_types[star] + total_tokens[star] + 1))

    for star in stars:
        scores[star] = smoothing_probs[star].get(reviewwords[0], default_prob[star])
        for i in range(1, len(reviewwords)):
            scores[star] *= smoothing_probs[star].get(reviewwords[i], default_prob[star])

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

def calculate_textblob(review):
    cur_review = TextBlob(' '.join(review))
    cur_sentiment = cur_review.sentiment.polarity

    # The only category with a negative polarity is 1_star
    if cur_sentiment < 0:
        return "1_star"
    
    # Find which the sentiment of this review is most similar to
    most_similar = None
    min_diff = None
    for star in sentiment_vals:
        if star == "1_star":
            continue
        cur_diff = abs(sentiment_vals[star] - cur_sentiment)
        if min_diff == None:
            min_diff = cur_diff
            most_similar = star
            continue
        if cur_diff < min_diff:
            min_diff = cur_diff 
            most_similar = star
            
    return most_similar

def calculate_accuracy(all_probs):
    true_results = {"nb": [], "smnb": [], "tb": []}
    pred_results = {"nb": [], "smnb": [], "tb": []}
    nbcorrect = {"1_star": 0, "2_star": 0, "3_star": 0, "4_star": 0, "5_star": 0}
    smnbcorrect = {"1_star": 0, "2_star": 0, "3_star": 0, "4_star": 0, "5_star": 0}
    tbcorrect = {"1_star": 0, "2_star": 0, "3_star": 0, "4_star": 0, "5_star": 0}

    star_list = ["1_star", "2_star", "3_star", "4_star", "5_star"]
    all_files = glob.glob("./json_data/tokenize/first_10k/test/*")
    for file in all_files:
        star_fd = open(file, 'r', encoding='utf-8')
        data = json.loads(star_fd.read())

        random.shuffle(data)

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

            # Apply each classifier and append to arrays of correct and predicted results
            # NB Calculation
            nbstar = naive_bayes(testwords, all_probs)
            if real_star == nbstar:
                nbcorrect[real_star] += 1

            true_results["nb"].append(real_star)
            pred_results["nb"].append(nbstar)

            # SMNB Calculation
            smnbstar = smooth_naive_bayes(testwords)
            if real_star == smnbstar:
                smnbcorrect[real_star] += 1

            true_results["smnb"].append(real_star)
            pred_results["smnb"].append(smnbstar)

            # TextBlob Calculation            
            tbstar = calculate_textblob(testwords)

            if real_star == tbstar:
                tbcorrect[real_star] +=1 

            true_results["tb"].append(real_star)
            pred_results["tb"].append(tbstar)

        star_fd.close()

    # NB Confusion Matrix
    nb_cm = confusion_matrix(true_results["nb"], pred_results["nb"], labels=["1_star", "2_star", "3_star", "4_star", "5_star"])
    nb_precision = np.mean(np.diag(nb_cm) / np.sum(nb_cm, axis = 0))
    nb_recall = np.mean(np.diag(nb_cm) / np.sum(nb_cm, axis = 1))

    # SMNB Confusion Matrix
    smnb_cm = confusion_matrix(true_results["smnb"], pred_results["smnb"], labels=["1_star", "2_star", "3_star", "4_star", "5_star"])
    smnb_recall = np.mean(np.diag(smnb_cm) / np.sum(smnb_cm, axis = 1))
    smnb_precision = np.mean(np.diag(smnb_cm) / np.sum(smnb_cm, axis = 0))

    ConfusionMatrixDisplay(confusion_matrix=smnb_cm, display_labels=["1_star", "2_star", "3_star", "4_star", "5_star"]).plot()
    # plt.show()
    # TextBlob Confusion Matrix
    tb_cm = confusion_matrix(true_results["tb"], pred_results["tb"], labels=["1_star", "2_star", "3_star", "4_star", "5_star"])
    tb_recall = np.mean(np.diag(tb_cm) / np.sum(tb_cm, axis = 1))
    tb_precision = np.mean(np.diag(tb_cm) / np.sum(tb_cm, axis = 0))

    # NB Results
    for star in nbcorrect:
        print("Naive Bayes Accuracy: ", star, (nbcorrect[star]/400))
    print("Naive Bayes Precision: ", nb_precision)
    print("Naive Bayes Recall: ", nb_recall)
    print("Naive Bayes f1: ", 2*((nb_precision * nb_recall)/(nb_precision + nb_recall)))


    # SMNB Results
    for star in nbcorrect:
        print("Smoothing Naive Bayes Accuracy: ", star, (smnbcorrect[star]/400))
    print("Smoothing Naive Bayes Precision: ", smnb_precision)
    print("Smoothing Naive Bayes Recall: ", smnb_recall)
    print("Smoothing Naive Bayes f1: ", 2*((smnb_precision * smnb_recall)/(smnb_precision + smnb_recall)))
   
    # TextBlob Results
    for star in nbcorrect:
        print("TextBlob Accuracy: ", star, (tbcorrect[star]/400))
    print("TextBlob Precision: ", tb_precision)
    print("TextBlob Recall: ", tb_recall)
    print("TextBlob f1: ", 2*((tb_precision * tb_recall)/(tb_precision + tb_recall)))

# MAIN SECTION #
get_training_data()
find_sentiment_range()
all_probs = calculate_nb_probabilities()
calculate_accuracy(all_probs)