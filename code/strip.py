import sys

inDirectory = ""
outDirectory = ""

inputname = sys.argv[1]
outputname = sys.argv[2]

output = open(outputname, "w")

for line in open(inputname, "r"):
	newline = line[:-1]
	tokens = newline.split()
	for i in range(len(tokens)):
		tokens[i] = strip_control_characters(tokens[i])
	# newline = strip_control_characters(line)
	# newline = newline + "\n"
	towrite = ""
	for t in tokens:
		towrite += t + " "
	towrite = towrite[:-1] + "\n"
	output.write(towrite)
	
output.close()

def strip_control_characters(input):
	if input:
		import re

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

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "d:o:")
		for opt, args in opts:
			if opt == "-d":
				global inDirectory
				inDirectory = args
			elif opt == "-o":
				global outDirectory
				outDirectory = 
	except getopt.GetoptError:
		usage()
		sys.exit(2)

if __name__ == '__main__':
	if len(sys.argv) < 5:
		usage()
		sys.exit()
	main(sys.argv[1:])
	strip()
