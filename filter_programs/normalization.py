import glob
import json
import nltk
from nltk.corpus import stopwords

def allfiles(route):
    allpos = glob.glob(route)

thesewords = list()
stoplist = stopwords.words('english')
stoplist.append(".")
stoplist.append(",")
stoplist.append("!")
stoplist.append("?")
stoplist.append("”")
stoplist.append("“")
stoplist.append(";")
stoplist.append("’")
stoplist.append("n't")
stoplist.append("'s")
stoplist.append("would")
stoplist.append("like")
def normalize(file, newfile):
     with open(file, "r") as f:
        review = json.load(f)
        lowercase =[]
        for r in review:
            alltokens = nltk.word_tokenize(r['text'])
            outfile = open("../json_data/top3/top3_user_1/" + newfile + ".json", "w")
            for word in alltokens:
                lowercase.append(word.lower())
            for w in lowercase:
                if w not in stoplist:
                    thesewords.append(w)
        json.dump(thesewords, outfile, indent=2)
        fdist = nltk.FreqDist(thesewords)
        print(fdist.most_common(10))
#normalize("../json_data/first_10k/test/first_10k_1_star_test.json", "normalize1")
normalize("../json_data/top3/top3_user_1/user_1_star_1.json", "normalized_1_star")

