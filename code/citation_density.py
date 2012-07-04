#!/usr/bin/python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parseString
import unicodedata
import nltk
import HTMLParser
import sys
import re

contextfile = "/Users/lwheng/Downloads/fyp/context.txt"

opencontextfile = open(contextfile, "r")
for l in opencontextfile:
	context = l.strip().split("!=")[1]
	if re.match(r'<context.*', context):
		dom = parseString(context)
		linesContext = dom.getElementsByTagName('context')[0].firstChild.data
		# citStr = dom.getElementsByTagName('context')[0].attributes['citStr']
		# print citStr.value
		linesContext = unicodedata.normalize('NFKD', linesContext).encode('ascii','ignore')
		query = nltk.word_tokenize(linesContext)
		query_display = ""
		for i in query:
			query_display = query_display + " " + i
		
		regex = r"(((\w+)\s*,?\s*(et al.?)?|(\w+ and \w+))\s*,?\s*(\(?\s?\d{4}\s?\)?)|\[\s*(\w+)\s*\]|\[\s(\w+\d+)\s\]|[\[|\(]\s(\d+\s?,\s?)*(\d+)\s[\]|\)]|\(\s*[A-Z]\w+\s*\)|\[\s(\w+\s,?\s?)+\])"
		# regex = r"\[\s*(\w+,?\s*)+\s*\]|\((\w+\;?\s*)+\)|,?\s*(et al.)\s*,?\s*\(?\s*\d{4}\s*\)?|[A-Z]\w+\s*,?\s*\(?\s*\d{4}\s*\)?"
		obj = re.findall(regex, query_display)
		if len(obj) == 0:
			print context
			print
			print query_display
			print 
			print obj
			sys.exit()
		else:
			citation_density = float(len(obj)) / float(len(query))
			# print query_display
			print str(len(obj)) + "/" + str(len(query)) + " = " + str(citation_density * 100)