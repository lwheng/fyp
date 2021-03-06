#!/opt/local/bin/python
# -*- coding: utf-8 -*-

import nltk
import cPickle as pickle

pickleFile = "/Users/lwheng/Downloads/fyp/contextCollection.pickle"

col = pickle.load(open(pickleFile,"r"))
print "Pickle loaded"
contextCollection_idf_hash = {}
tokens = col.vocab().keys()
print len(tokens)
thehash = {}
for text in col._texts:
	print text
	for t in tokens:
		if t in text:
			if t not in thehash:
				thehash[t] = 0
			thehash[t] += 1
print thehash

# for t in col.vocab().keys():
# 	print t
# 	contextCollection_idf_hash[t] = col.idf(t)

# pickle.dump(open('contextCollection_idf.pickle', 'w'))