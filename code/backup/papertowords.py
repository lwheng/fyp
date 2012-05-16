#!/opt/local/bin/python
# -*- coding: utf-8 -*-

import sys

if len(sys.argv) < 3:
	print "Error! Please specify input and output file."
	print "Usage: python " + sys.argv[0] + " <input file> <output file>"
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
for line in input:
	words = line.split()
	for w in words:
		# print "w is " + w
		word = w.lower()ls
		
		bag.append(word)
		
voc = FreqDist(bag)
for w in voc:
	word = w + "\n"
	# print w
	output.write(word)
	
input.close()
output.close()
		
