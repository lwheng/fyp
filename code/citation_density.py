#!/usr/bin/python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parseString
import unicodedata
import nltk
import HTMLParser
import sys
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer

sentenceTokenizer = PunktSentenceTokenizer()

contextfile = "/Users/lwheng/Downloads/fyp/context.txt"

opencontextfile = open(contextfile, "r")
reg = []
reg.append(r"\(\s?(\d{1,3})\s?\)")
reg.append(r"\(\s?(\d{4})\s?\)")
reg.append(r"\(\s?(\d{4};?\s?)+\s?")
reg.append(r"\[\s?(\d{1,3}\s?,?\s?)+\s?\]")
reg.append(r"\[\s?([\w-],?\s?)+\s?\]")
reg.append(r"([A-Z][A-Za-z-]+\s?,?\s?(\s(and|&)\s)?)+\s?,?\s?(et al\.?)?\s?,?\s?(\(?(\d{4})\)?)")

regex = ""
for i in range(len(reg)):
  regex += reg[i] + "|"
regex = re.compile(regex[:-1])
for l in opencontextfile:
  context = l.strip().split("!=")[1]
  if re.match(r'<context.*', context):
    dom = parseString(context)
    linesContext = dom.getElementsByTagName('context')[0].firstChild.data
    # citStr = dom.getElementsByTagName('context')[0].attributes['citStr']
    # print citStr.value
    linesContext = unicodedata.normalize('NFKD', linesContext).encode('ascii','ignore')

    # regex = r"(((\w+)\s*,?\s*(et al.?)?|(\w+ and \w+))\s*,?\s*(\(?\s?\d{4}\s?\)?)|\[\s*(\w+)\s*\]|\[\s(\w+\d+)\s\]|[\[|\(]\s(\d+\s?,\s?)*(\d+)\s[\]|\)]|\(\s*[A-Z]\w+\s*\)|\[\s(\w+\s,?\s?)+\])"

    query = nltk.word_tokenize(linesContext)
    query_display = ""
    for i in query:
      query_display = query_display + " " + i
    objNorm = re.findall(regex, query_display)
    if len(objNorm) == 0:
      print objNorm
      print query_display
      print
      sys.exit()
    else:
      print query_display
      print

    # queryLines = sentenceTokenizer.tokenize(linesContext)
    # numOfCitations = 0
    # for l in queryLines:
    #   obj = re.findall(regex,l)
    #   # print l + " =====> " + str(len(obj))
    #   numOfCitations += len(obj)
    # # sys.exit()













