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

# infile = input.read()
# words = infile.split()

bag = []
count = 1
for line in input:
	# print "Start of line " + str(count)
	words = line.split()
	for w in words:
		if len(w) > 3:
			bag.append(w)
	# print "End of line " + str(count)
	count = count + 1
		
voc = FreqDist(bag)
for w in voc:
	word = w + "\n"
	# print w
	output.write(word)
	
input.close()
output.close()
		