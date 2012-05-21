# This script takes in a set of documents or a text file and generates
# the Document Frequency table.

# Needs
# 1. vocab
# 2. Document directory or a text file

import os
import sys
import string
import getopt
import myUtils

from datetime import datetime

date = str(datetime.now().date())
time = str(datetime.now().time())

inputFile = ""
fileDirectory = ""
vocabFile = ""
dftableFile = "/Users/lwheng/Downloads/fyp/dftable-(" + date + "-" + time + ").txt"

dftableDict = {}

def loadVocabFile():
    global vocabFile
    global dftableDict
    if len(vocabFile) == 0:
        usage()
        sys.exit(2)
    openvocab = open(vocabFile,"r")
    for l in openvocab:
        line = l[:-1]
        dftableDict[line] = 0
    openvocab.close()

def dftable():
    global fileDirectory
    global dftableFile
    global inputFile

    if len(inputFile) == 0:
        for dirname, dirnames, filenames in os.walk(fileDirectory):
            for filename in filenames:
                thefile = str(os.path.join(dirname, filename))
                openthefile = open(thefile,"r")
                fileDict = {}
                for l in openthefile:
                    line = myUtils.removespecialcharacters(l)
                    line = line.lower()
                    tokens = line.split()
                    for t in tokens:
                        toadd = ""
                        if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
                            toadd = t
                        elif (myUtils.removepunctuation(t)).isalnum():
                            toadd = myUtils.removepunctuation(t)

                        if len(toadd) != 0:
                            if toadd not in fileDict:
                                fileDict[toadd] = 0
                            fileDict[toadd] += 1
                for key in fileDict:
                    if key in dftableDict:
                        dftableDict[key] += 1
                openthefile.close()
        openoutput = open(dftableFile,"w")
        for k in dftableDict:
            towrite = k + "\t" + str(dftableDict[k]) + "\n"
            openoutput.write(towrite)
        openoutput.close()
    elif len(inputFile) > 0:
        openthefile = open(inputFile,"r")
        fileDict = {}
        for l in openthefile:
            line = myUtils.removespecialcharacters(l)
            line = line.lower()
            tokens = line.split()
            for t in tokens:
                toadd = ""
                if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
                    toadd = t
                elif (myUtils.removepunctuation(t)).isalnum():
                    toadd = myUtils.removepunctuation(t)

                if len(toadd) != 0:
                    if toadd not in fileDict:
                        fileDict[toadd] = 0
                    fileDict[toadd] += 1
        for key in fileDict:
            if key in dftableDict:
                dftableDict[key] += 1
        openthefile.close()
        openoutput = open(dftableFile,"w")
        for k in dftableDict:
            towrite = k + "\t" + str(dftableDict[k]) + "\n"
            openoutput.write(towrite)
        openoutput.close()

def usage():
    print "USAGE: python " + sys.argv[0] +" -v <vocab file> [-d <fileDirectory> | -s <text file>]"
    print "Default output location is Downloads, dftable-(timestamp).txt"
    print "To specify output, add this: -o <output filename>"

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "v:d:o:s:")
        for opt, args in opts:
            if opt == "-d":
                global fileDirectory
                fileDirectory = args
            elif opt == "-v":
                global vocabFile
                vocabFile = args
            elif opt == "-o":
                global dftableFile
                dftableFile = args
            elif opt == "-s":
                global inputFile
                inputFile = args
    except getopt.GetoptError:
        usage()
        sys.exit(2)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        sys.exit()
    main(sys.argv[1:])
    loadVocabFile()
    dftable()
    
