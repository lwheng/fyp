import nltk.text
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

# Define where is the root directory of the corpus
root = "/Users/lwheng/Desktop/FilesCleaned/"
corpus = PlaintextCorpusReader(root, ".*\.txt")

freq = tf("the", corpus)
print "freq is " + str(freq)