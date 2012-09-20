#!/opt/local/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import re
import unicodedata
import cPickle as pickle

# Compiles all the paper titles for easy reference

metadataDir = "/Users/lwheng/Downloads/fyp/metadata/"
authorsHash = {}
files = os.listdir(metadataDir)
for f in files:
	filename = metadataDir + f
	openfile = open(filename,"r")
	reOpenPaper = r"<paper id=\"(.*)\">"
	reClosePaper = r"</paper>"
	reTitle = r"<title>(.*)</title>"
	reAuthor = r"<author>(.*)</author>"
	openauthor = False
	for l in openfile:
		matchOpen = re.findall(reOpenPaper,l.strip())
		matchClose = re.findall(reClosePaper,l.strip())
		if matchOpen:
			authors = []
			openauthor = True
			paperid = matchOpen[0]
		if matchClose:
			openauthor = False
			# print out
			# print f[0:3] + "-" + paperid + "==>" + str(authors)
			authorsHash[f[0:3] + "-" + paperid] = authors
		if openauthor:
			matchAuthor = re.findall(reAuthor,l.strip())
			if matchAuthor:
				# print f[0:3] + "-" + paperid + "==>" + matchTitle[0]
				# authorname = unicodedata.normalize('NFKD', unicode(matchAuthor[0])).encode('ascii','ignore')
				authors.append(matchAuthor[0])
	openfile.close()

pickle.dump(authorsHash, open("paperAuthors.pickle", "wb"))