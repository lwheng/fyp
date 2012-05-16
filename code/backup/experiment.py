# This is the file that will compute similarity

# Consider the citation, [Paper A]==>[Paper B]
# What information do we need?
# 1. The search query - This would be the partial sentence from Paper A
# 2. The search domain - Paper
# 3. Entire bank of vocab

# Input/Output:
# Input: Query, Domain, Vocab, DFTable
# Output: Percentage of similarity

# USAGE:
# python compute.py <queryfile> <domainfile> <vocabfile> <dftablefile>

# LOG:
# Use Python List for Vectors
# v = [1,2,3,...], len(v) = no. of dimensions
# Yet to decide type of domain and vocab

import sys
import string
import math
import getopt
import nltk
from nltk.probability import FreqDist
from nltk.corpus import PlaintextCorpusReader

queryfile = ""
domainfile = ""
vocabfile = ""
dftablefile = ""

queryarr = []
domain = ""
vocab = ""
dftable = {}

def addAllFilesToMemory():
	root = "/Users/lwheng/Desktop/FilesCleaned/"
	corpus = PlaintextCorpusReader(root, ".*\.txt")

	# # Read the query into memory
	# readquery = open(queryfile, "r")
	# query = ""
	# for line in readquery:
	# 	query = line
	# readquery.close()
	# # query is a line of words, let's split them for future use
	# global queryarr
	# queryarr = query.split()

	# # Read the domain into memory
	# domain = ""
	# readdomain = open(domainfile, "r")
	# for line in readdomain:
	# 	# add to memory
	# 	print "Read domain"
	# readdomain.close()

	# # Read vocab into memory
	# vocab = "" 
	# readvocab = open(vocabfile, "r")
	# for line in readvocab:
	# 	# add to memory
	# 	print "Read vocab"
	# readvocab.close()

	# Read DFTable into memory
	readdftable = open(dftablefile, "r")
	global dftable
	for line in readdftable:
		info = line.split("=====>")
		key = info[0]
		value = info[1]
		dftable[key] = value

def cosine(x):
	# Return the cosine of x radians
	return math.cos(x)

def log(x):
	# Return the natual log of x
	return math.log(x)

def dotproduct(v1, v2):
	# Computes dot product of 2 vectors
	# len(v1) must be equal to len(v2) (You don't say!)
	if (len(v1) != len(v2)):
		return False
	output = 0
	for i in range(len(v1)):
		output = output + (v1[i]*v2[i])
	return output

def magnitude(v):
	# Computes Euclidean Length of vector
	output = 0
	sumofsquares = 0
	for i in v:
		sumofsquares = sumofsquares + i**2
	output = math.sqrt(sumofsquares)
	return output

def cosinesim(v1, v2):
	# Computes cosine similarity
	return dotproduct(v1,v2)/(magnitude(v1)*magnitude(v2))

def tf(term, document):
	print "hello"


def df(term):
	if (dftable[term]):
		return dftable[term]
	else:
		return 0

def idf(N, df):
	return log(N/df)

def sim(d1, d2):
	# d1 : query
	# d2 : domain
	# Computes similarity
	# To be completed
	print "sim"

def usage():
	print "USAGE: python " + sys.argv[0] +" -q <queryfile> -d <domainfile> -v <vocabfile> -f <dftablefile>"

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "q:d:v:f:")
        for opt, args in opts:
            if opt == "-q":
                global queryfile
                queryfile = args
            elif opt == "-d":
                global domainfile
                domainfile = args
            elif opt == "-v":
                global vocabfile
                vocabfile = args
            elif opt == "-f":
            	global dftablefile
            	dftablefile = args
    except getopt.GetoptError:
	    usage()
	    sys.exit(2)

if __name__ == '__main__':
	main(sys.argv[1:])
	addAllFilesToMemory()
	print df("the")







