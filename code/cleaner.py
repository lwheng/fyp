import sys
import string

if len(sys.argv) < 2:
	print "Fail"
	sys.exit()
	
input = open(sys.argv[1], "r")
output = open("ListOfFile", "w")

infile = input.read()
words = infile.split()

for w in words:
	if len(w) > 3:
		word = w + "\n"
		output.write(word)