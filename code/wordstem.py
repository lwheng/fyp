# INPUT: A set of documents
# OUTPUT: The set of documents, but with words lemmatized

import os
import sys
import string
import getopt
import myUtils

fileDirectory = ""

def stemmer():
	print "hello world"

def usage():
	print "USAGE: python " + sys.argv[0] +" -d <fileDirectory>"

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "d:o:")
		for opt, args in opts:
			if opt == "-d":
				global fileDirectory
				fileDirectory = args
	except getopt.GetoptError:
		usage()
		sys.exit(2)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		usage()
		sys.exit()
	main(sys.argv[1:])
	stemmer()
