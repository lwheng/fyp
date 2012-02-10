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

for line in open(inputname, "r"):
	# print line
	print "Start of line"
	newline = strip_control_characters(line)
	newline = newline + "\n"
	output.write(newline)
	print "End of line"
	
inputname.close()
outputname.close()