# Program to read find NB probability for 1-5 star reviews 
# Based off nb.py file from Pset 1 
from os import curdir
from threading import current_thread
from nltk import FreqDist
from nltk.corpus import stopwords
import glob
import math
import re
from textblob import TextBlob
import json

# Stopwords
stops = stopwords.words('english')
punct = ["'", ".", ",", '"', ";", "?", "!", "-", "-", "(", ")", "*", "/", ":", "~"]

# Store all content words in each review
stars = {"1_star": [], "2_star": [], "3_star": [], "4_star": [], "5_star": [],}

# Store the values obtained from Textblob from training data
sentiment_vals = {"1_star": None, "2_star": None, "3_star": None, "4_star": None, "5_star": None}

# Store the NB scores obtained from from training data
train_scores = {"1_star" : None, "2_star" : None, "3_star" : None, "4_star" : None, "5_star" : None}

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
        # !!! print(cur_freq.most_common(5))
        # Find nb probabilities
        for word in cur_freq:
            word_count = cur_freq[word]
            cur_prob = math.log(word_count/total_count)
            all_probs[star][word] = cur_prob

    return all_probs

def get_nb_baseline(stars, train_scores):
    # Get Baseline Stars 
    # Get the values we will compare the test data with to categorize reviews 
    defaultprob = math.log(0.0000000000001)
    for star in all_probs:
        print(len(stars[star]))
        train_scores[star] = all_probs[star].get(stars[star][0], defaultprob)
        for i in range(1, len(stars[star])):
            train_scores[star] += all_probs[star].get(stars[star][i], defaultprob)
        train_scores[star] = train_scores[star]/len(stars[star])
    print("train_scores", train_scores)
    
    return train_scores

def naive_bayes(reviewwords, all_probs, train_scores):
    defaultprob = math.log(0.0000000000001)
    
    test_scores = {
        "1_star" : None,
        "2_star" : None,
        "3_star" : None,
        "4_star" : None,
        "5_star" : None
    }

    for star in all_probs:
        print(len(reviewwords))
        test_scores[star] = all_probs[star].get(reviewwords[0], defaultprob)
        for i in range(1, len(reviewwords)):
            test_scores[star] += all_probs[star].get(reviewwords[i], defaultprob)
        test_scores[star] = train_scores[star]/len(reviewwords)
    print("test_scores", test_scores)
        
    # for score in scores:
    #     print(score)    
    #     print(scores[score])


    
    # ### POSITIVE SCORE
    # posscore = poswordprobs.get(reviewwords[0], defaultprob)
    # for i in range(1, len(reviewwords)):
    #     posscore += poswordprobs.get(reviewwords[i], defaultprob)

    # ### CALCULATE NEGATIVE SCORE
    # negscore = negwordprobs.get(reviewwords[0], defaultprob)
    # for i in range(1, len(reviewwords)):
    #     negscore += negwordprobs.get(reviewwords[i], defaultprob)

    # if (posscore - negscore) >  0:
    #     return "pos"

    # return "neg"

def calculate_accuracy(all_probs, train_scores):
    nbcorrect = 0
    smnbcorrect = 0
    tbcorrect = 0
    affcorrect = 0

    stars = ["1_star", "2_star", "3_star", "4_star", "5_star"]
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
        star_fd.close()

        # Find which review we are working with 1 star, 2 star etc.
        for star in stars:
            if star in file:
                real_star = star
                # If star found, break out of loop
                continue
        
        # Apply each classifier to review and check to see if correct
        if real_star == naive_bayes(testwords, all_probs, train_scores):
            nbcorrect += 1


# MAIN SECTION #
get_training_data()
find_sentiment_range()

all_probs = calculate_nb_probabilities()

train_scores = get_nb_baseline(stars, train_scores)
print(train_scores)
calculate_accuracy(all_probs, train_scores)