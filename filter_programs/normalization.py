import json
import urllib
import re
import nltk

stops = {"'d", "'ll", "'m", "'ve", "'t", "'s", "'re", "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "let's", "me", "mightn't", "more", "most", "mustn't", "my", "myself", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "should've", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "will", "with", "won't", "would", "wouldn", "wouldn't", "y'all", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", ",", ".", "!", "?", "'", '"', "I", "i"}

def normalize(file):
     with open(file, "r") as f:
        review = json.load(f)
        for r in review:
            thesewords = set()
            outfile = open("../json_data/first_10k/test/normalized_text.json", "w")
            words = r['text'].rstrip().split()
            for w in words:
                if w not in stops:
                    thesewords.add(w)
            json.dump(thesewords, outfile, indent=2)
normalize("../json_data/first_10k/test/first_10k_1_star_test.json")

