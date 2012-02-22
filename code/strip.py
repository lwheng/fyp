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
	
import sys

inputname = sys.argv[1]
outputname = sys.argv[2]

output = open(outputname, "w")

wc = 0
for line in open(inputname, "r"):
	# print line
	newline = line[:-1]
	tokens = newline.split("\t")
	for i in range(len(tokens)):
		tokens[i] = strip_control_characters(tokens[i])
	# newline = strip_control_characters(line)
	# newline = newline + "\n"
	if len(tokens) == 3:
		towrite = tokens[0] + "\t" + tokens[1] + "\t" + tokens[2] + "\n"
		output.write(towrite)
		wc = wc + 1
		print str(wc) + " of 12928923"
	
outputname.close()
