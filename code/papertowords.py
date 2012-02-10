#!/opt/local/bin/python
# -*- coding: utf-8 -*-

import sys

if len(sys.argv) < 3:
	print "Error! Please specify input and output file."
	print "Usage: python cleaner.py <input file> <output file>"
	sys.exit()

import string	
from nltk.probability import FreqDist
	
inputname = sys.argv[1]
outputname = sys.argv[2]

input = open(inputname, "r")
output = open(outputname, "w")

infile = input.read()
words = infile.split()

bag = []
for w in words:
	if len(w) > 3:
		bag.append(w)
		
voc = FreqDist(bag)
for w in voc:
	word = w + "\n"
	output.write(word)
	
input.close()
output.close()
		