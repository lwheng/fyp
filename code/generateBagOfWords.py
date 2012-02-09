import sys

if (len(sys.argv) < 2):
	print "Error: No input file specified"
	print "Usage: python generateBagOfWords.py [input file]"
	sys.exit()
	
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
import string
lm = WordNetLemmatizer()

input = open(sys.argv[1], "r")
output = open("ThisIsTheTempFileOneCannotMiss", "w")

infile = input.read()
words = infile.split()
newWords = []

for w in words:
	if len(w) > 0:
		newW = string.lower(w)
		newW = lm.lemmatize(newW)
		newWords.append(newW)
		
voc = FreqDist(newWords)

for w in voc:
	# word = w + " ==> " + str(voc[w]) + "\n"
	word = w + "\n"
	output.write(word)
	
input.close()
output.close()