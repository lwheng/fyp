import pickle
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import sys
import os
import unicodedata

wn = WordNetLemmatizer()
path = "/Users/lwheng/Downloads/fyp/pdfbox-0.72"

docs = []
docs_col = ""
for (path, dirs, files) in os.walk(path):
	if files:
		for f in files:
			filename = os.path.join(path, f)
			openfile = open(filename, "r")
			data = openfile.read()
			data = data.lower()
			tokens = nltk.word_tokenize(data)
			# subvocab = list(set(tokens))
			for i in range(len(tokens)):
				tokens[i] = wn.lemmatize(tokens[i])
			docs.append(nltk.Text(tokens))
docs_col = nltk.TextCollection(docs)
pickle.dump(docs_col, open('collection.p','wb'))