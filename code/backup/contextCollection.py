#!/opt/local/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import re
from xml.dom.minidom import parseString
import unicodedata
import nltk
import cPickle as pickle
from nltk.corpus import stopwords

parscitFileDir = "/Users/lwheng/Downloads/fyp/parscitxml500"
contexts = []
for (path, dirs, files) in os.walk(parscitFileDir):
	for f in files:
		# print f
		filename = os.path.join(parscitFileDir, f)
		openfile = open(filename,"r")
		data = openfile.read()
		openfile.close()
		dom = parseString(data)
		contextsTemp = dom.getElementsByTagName('context')
		for c in contextsTemp:
			print c.toxml()
			content = c.firstChild.data.lower()
			content = unicodedata.normalize('NFKD', content).encode('ascii','ignore')
			contexts.append(nltk.Text(nltk.word_tokenize(content)))

contextCollection = nltk.TextCollection(contexts)
# pickle.dump(contextCollection, open("contextCollection.pickle", "wb"))


