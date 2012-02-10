import sys

idfname = sys.argv[1]

idfhash = {}
for line in open(idfname, "r"):
	l = line[:-1]
	idfhash[l]=0
# All words are now in hash

