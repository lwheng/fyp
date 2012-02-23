# This script takes in a set of documents and generate the set of the words
# found in the documents.

import os
import sys
import getopt
from nltk.stem.wordnet import WordNetLemmatizer
from datetime import datetime

date = str(datetime.now().date())
time = str(datetime.now().time())

fileDirectory = ""
vocabFile = "/Users/lwheng/Desktop/vocab-(" + date + "-" + time + ").txt"

lmtzr = WordNetLemmatizer()

def vocab():
    global fileDirectory
    global vocabFile
    global lmtzr
    
    for dirname, dirnames, filenames in os.walk(fileDirectory):
        for filename in filenames:
            thefile = str(os.path.join(dirname, filename))
            openthefile = open(thefile, "r")

def usage():
	print "USAGE: python " + sys.argv[0] +" -d <fileDirectory>"
	print "Default output location is Desktop, vocab-(timestamp).txt"
	print "To specify output, add this: -o <output filename>"

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:o:")
        for opt, args in opts:
            if opt == "-d":
                global fileDirectory
                fileDirectory = args
            elif opt == "-o":
                global vocabFile
                vocabFile = args
    except getopt.GetoptError:
    	usage()
    	sys.exit(2)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        sys.exit()
    main(sys.argv[1:])
    vocab()









