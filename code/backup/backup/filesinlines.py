import sys
import string

listname = sys.argv[1]
outputfile = sys.argv[2]

openlistname = open(listname, "r")
output = open(outputfile,"w")

for f in openlistname:
	filename = f[:-1]
	count = 1
	paper = string.split(string.split(filename, "/")[-1], ".")[0]
	print paper
	openfile = open(filename, "r")
	for l in openfile:
		line = l
		towrite = paper + "-" + str(count) + " : " + line
		output.write(towrite)
		count = count + 1
	
	
