import math
from nltk import FreqDist
import glob

poswords = []
negwords = []

poswordprobs = {}
negwordprobs = {}

def read_in_training_data():


    ## Read in all positive reviews
    ## We create a set of unique words for each review. 
    ## We then add that set of words as a list to the master list of positive words.
    positivewords = []
    allpos = glob.glob("pos/*")
    for filename in allpos:
        f = open(filename)
        thesewords = set()
        for line in f:
            words = line.rstrip().split()
            for w in words:
                if w not in stops:
                    thesewords.add(w)
        f.close()
        positivewords.extend(list(thesewords))
    
    print(len(positivewords), "positive tokens found!")
    print(len(set(positivewords)), "positive types found!")
    
    
    ## Read in all negative reviews
    ## We create a set of unique words for each review.
    ## We then add that set of words as a list to the master list of negative words.
    negativewords = []
    allneg = glob.glob("neg/*")
    for filename in allneg:
        f = open(filename)
        thesewords = set()
        for line in f:
            words = line.rstrip().split()
            for w in words:
                if w not in stops:
                    thesewords.add(w)
        f.close()
        negativewords.extend(list(thesewords))
    
    print(len(negativewords), "negative tokens found!")
    print(len(set(negativewords)), "negative types found!")
    return(positivewords, negativewords)
def calculate_nb_probabilities():

    poswordprobs = {}
    negwordprobs = {}

    pdist = FreqDist(poswords)
    
    ndist = FreqDist(negwords)

    print("Naive bayes probabilities...")
    print("positive...")
    for word in pdist:
        posprobability = pdist.get(word) / len(poswords)
        poswordprobs[word] = math.log(posprobability)
    print("negative...")
    for word in ndist:
        negprobability = ndist.get(word) / len(negwords)
        negwordprobs[word] = math.log(negprobability)
    return (poswordprobs, negwordprobs)
    
def naiveBayes():
    defaultprob = math.log(0.0000000000001)
    
    posscore = poswordprobs.get(reviewwords[0], defaultprob)
    for i in range(1, len(reviewwords)):
        posscore += poswordprobs.get(reviewwords[i], defaultprob)

    negscore = negwordprobs.get(reviewwords[0], defaultprob)
    for i in range(1, len(reviewwords)):
        negscore += negwordprobs.get(reviewwords[i], defaultprob)

    if (posscore - negscore) >  0:
        return "pos"

    return "neg"