# This script takes in a set of documents and generate the set of the words
# found in the documents.

import os
import sys
import string
import getopt
from nltk.stem.wordnet import WordNetLemmatizer
from datetime import datetime

date = str(datetime.now().date())
time = str(datetime.now().time())

fileDirectory = ""
vocabFile = "/Users/lwheng/Desktop/vocab-(" + date + "-" + time + ").txt"

lmtzr = WordNetLemmatizer()

def hyphenated(word):
    tokens = word.split("-")
    for t in tokens:
        if not t.isalnum():
            return False
    return True

def apos(word):
    tokens = word.split("'")
    if len(tokens) != 2:
        return False
    for t in tokens:
        if not t.isalnum():
            return False
    return True

def removepunctuation(word):
    output = ""
    for w in word:
        if not w in string.punctuation:
            output += w
    return output

def vocab():
    global fileDirectory
    global vocabFile
    global lmtzr

    vocabDict = {}
    
    for dirname, dirnames, filenames in os.walk(fileDirectory):
        for filename in filenames:
            thefile = str(os.path.join(dirname, filename))
            openthefile = open(thefile, "r")
            for l in openthefile:
                line = l[:-1]
                line = line.lower()
                tokens = line.split()
                for t in tokens:
                    toadd = ""
                    if t.isalnum() or hyphenated(t) or apos(t):
                        toadd = t
                    elif removepunctuation(t).isalnum():
                        toadd = removepunctuation(t)

                    if len(toadd) != 0:
                        if toadd not in vocabDict:
                            vocabDict[toadd] = 0
                        vocabDict[toadd] += 1

    openvocabfile = open(vocabFile,"w")
    for k in vocabDict:
        towrite = k + "\n"
        openvocabfile.write(towrite)

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









