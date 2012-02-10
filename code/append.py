import sys

inputname = sys.argv[1]
outputname = sys.argv[2]
output = open(outputname, "w")

for line in open(inputname, "r"):
	newline = line[:-1]
	newline = newline + "==>" + "\n"
	output.write(newline)