import os
import sys
import string   
import getopt
import re
import myUtils

inDirectory = ""
outDirectory = ""

def strip_control_characters(input):
	if input:
		# unicode invalid characters
		RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
						u'|' + \
						u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
						(unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
						unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
						unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
						)
		input = re.sub(RE_XML_ILLEGAL, "", input)

		# ascii control characters
		input = re.sub(r"[\x01-\x1F\x7F]", "", input)
	return input

def strip():
	global inDirectory
	global outDirectory

	if not os.path.isdir(outDirectory):
		os.makedirs(outDirectory)

	for dirname, dirnames, filenames in os.walk(inDirectory):
		for f in filenames:
			inputName = str(os.path.join(inDirectory, f))
			outputName = str(os.path.join(outDirectory, f))

			readinput = open(inputName,"r")
			writeoutput = open(outputName,"w")
			for l in readinput:
				tokens = l.split()
				towrite = ""
				for t in tokens:
					towrite += strip_control_characters(t) + " "
				towrite = towrite[:-1] + "\n"
				writeoutput.write(towrite)
			readinput.close()
			writeoutput.close()


def usage():
	print "USAGE: python " + sys.argv[0] +" -d <inDirectory> -o <outDirectory>"

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "d:o:")
		for opt, args in opts:
			if opt == "-d":
				global inDirectory
				inDirectory = args
			elif opt == "-o":
				global outDirectory
				outDirectory = args
	except getopt.GetoptError:
		usage()
		sys.exit(2)

if __name__ == '__main__':
	if len(sys.argv) < 5:
		usage()
		sys.exit()
	main(sys.argv[1:])
	strip()
