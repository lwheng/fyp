# This file is written to get rid of non-ascii character in a text file

import sys

if len(sys.argv) < 3:
	print "Error! Please specify input and output file."
	print "Usage: python cleaner.py <input file> <output file>"
	sys.exit()

import string	
	
inputname = sys.argv[1]
outputname = sys.argv[2]

def ascii_char(c):
	return (ord(c) < 128)

def is_ascii(s):
	return all(ord(c) < 128 for c in s)

input = open(inputname, "r")
output = open(outputname, "w")

for line in input:
	newline = ""
	if is_ascii(line):
		newline = line
		output.write(newline)
	else:
		for c in line:
			if ascii_char(c):
				newline = newline + c
		output.write(newline)


# infile = input.read()
# words = infile.split()
input.close()
output.close()
		