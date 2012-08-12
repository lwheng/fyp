from xml.dom.minidom import parseString
import unicodedata
import nltk
import sys
import re
import os
import math
import numpy
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.metrics import distance

class dist:
	def levenshtein(self, a, b):
		return distance.edit_distance(a, b)

	def levenshteinRatio(self, a, b):
		lensum = float(len(a) + len(b))
		if lensum == 0.0:
			return 1.0
		ldist = self.levenshtein(a,b)
		return (lensum - ldist) / lensum

	def jaccard(self, inputA, inputB):
		# Returns jaccard index. Smaller the more better
		a = inputA.lower()
		b = inputB.lower()
		return distance.jaccard_distance(set(a.split()), set(b.split()))

class pickler:
	# Pickle files
	pickle_paperTitles = "/Users/lwheng/Downloads/fyp/paperTitles.pickle"
	pickle_paperAuthors = "/Users/lwheng/Downloads/fyp/paperAuthors.pickle"
	titles = {}
	authors = {}
	# pickle_contextCollection = "/Users/lwheng/Downloads/fyp/contextCollection.pickle"

	def __init__(self, paperTitles="/Users/lwheng/Downloads/fyp/paperTitles.pickle", paperAuthors="/Users/lwheng/Downloads/fyp/paperAuthors.pickle"):
		self.pickle_paperTitles = paperTitles
		self.pickle_paperAuthors = paperAuthors
		self.titles = self.fetchTitles()
		self.authors = self.fetchAuthors()

	def fetchTitle(self, paperID):
		return self.titles[paperID]

	def fetchTitles(self):
		tempTitle = {}
		tempTitle = pickle.load(open(self.pickle_paperTitles, "rb"))
		return tempTitle

	def fetchAuthors(self):
		tempAuthors = {}
		tempAuthors = pickle.load(open(self.pickle_paperAuthors, "rb"))
		# print "Loaded authors"
		return tempAuthors

	def fetchContexts(self, cite_key):
		info = cite_key.split("==>")
		citing = info[0]
		cited = info[1]

		titleToMatch = self.titles[cited]

		citingFile = "/Users/lwheng/Downloads/fyp/parscitxml500/" + citing + "-parscit.xml"
		openciting = open(citingFile,"r")
		data = openciting.read()
		openciting.close()
		dom = parseString(data)
		citations = dom.getElementsByTagName('citation')
		tags = ["title", "note", "booktitle", "journal"]
		titleTag = []
		index = 0
		bestIndex = 0
		minDistance = 314159265358979323846264338327950288419716939937510
		for i in range(len(citations)):
			c = citations[i]
			valid = c.getAttribute('valid')
			if valid == "true":
				titleTag = []
				index = 0
				while titleTag == []:
					titleTag = c.getElementsByTagName(tags[index])
					index += 1
				title = titleTag[0].firstChild.data
				title = unicodedata.normalize('NFKD', title).encode('ascii','ignore')
				thisDistance = levenshtein(title, titleToMatch)
				if thisDistance < minDistance:
					minDistance = thisDistance
					bestIndex = i
		return citations[bestIndex]