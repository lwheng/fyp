import nltk.text
from nltk.corpus import PlaintextCorpusReader
from nltk.text import TextCollection

# Define where is the root directory of the corpus
root = "/Users/lwheng/Desktop/FilesCleaned/"
corpus = PlaintextCorpusReader(root, ".*\.txt")

aclarc = TextCollection(corpus)

# freq = tf("the", corpus)
# print "freq is " + str(freq)