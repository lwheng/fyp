#!/usr/bin/python
# -*- coding: utf-8 -*-

# To determine the type of the citation.
# If specific, we wish to determine the location of the cited information

# 1. Using Citation Density 
# 2. Using Cosine Similarity

from xml.dom.minidom import parseString
import unicodedata
import nltk
import HTMLParser
import sys
import re
import math
import numpy
from nltk.corpus import stopwords

CHUNK_SIZE = 10
citationTypes = ['General', 'Specific', 'Undetermined']

cite_key = "W03-0415==>P99-1016"
context1 = """<context citStr=\"Caraballo (1999)\" endWordPosition=\"593\" position=\"3648\" startWordPosition=\"592\">. In Section 4, we show how correctly extracted relationships can be used as seed-cases to extract several more relationships, thus improving recall; this work shares some similarities with that of Caraballo (1999). In Section 5 we show that combining the techniques of Section 3 and Section 4 improves both precision and recall. Section 6 demonstrates that 1Another possible view is that hyponymy should only re</context>"""

query = []
domain = []
docs = []

display_query = ""

col_query = []
col_docs = []

fd_query = []

vocab = []

def citDensity(context):
  global query
  global display_query
  # context is in form of  "<context ... > ... </context>"
  dom = parseString(context)
  contextValue = dom.getElementsByTagName('context')[0].firstChild.data
  contextValue = unicodedata.normalize('NFKD', contextValue).encode('ascii','ignore')

  query = nltk.word_tokenize(contextValue)
  display_query = ""
  for i in query:
    display_query = display_query + " " + i

  regex = r"(((\w+)\s*,?\s*(et al.?)?|(\w+ and \w+))\s*,?\s*(\(?\s?\d{4}\s?\)?)|\[\s*(\w+)\s*\]|\[\s(\w+\d+)\s\]|[\[|\(]\s(\d+\s?,\s?)*(\d+)\s[\]|\)]|\(\s*[A-Z]\w+\s*\)|\[\s(\w+\s,?\s?)+\])"
  obj = re.findall(regex,display_query)

  # # Type:
  # 0: General
  # 1: Specific
  # 2: Undetermined
  if len(obj) == 1:
    return 1
  elif len(obj) > 1:
    return 0
  else:
    return 2

def cosineSimilarity(cite_key, context):
  global query
  global display_query
  global domain
  global docs
  global CHUNK_SIZE
  global vocab

  # Citing Paper
  fd_query = nltk.FreqDist(query)
  col_query = nltk.TextCollection([nltk.Text(query)])

  # Cited Paper
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]
  citedpaper = "/Users/lwheng/Downloads/fyp/pdfbox-0.72/" + cited[0] + "/" + cited[0:3] + "/" + cited + ".txt"
  domain = []
  try:
    openfile = open(citedpaper,"r")
    for l in openfile:
      domain.append(l.strip())
    openfile.close()
  except IOError as e:
    print e
  docs = []
  for i in xrange(0, len(domain), CHUNK_SIZE/2):
    sublist = domain[i:i+CHUNK_SIZE]
    temp = ""
    for s in sublist:
      temp = temp + " " + s
    docs.append(nltk.Text(nltk.word_tokenize(temp)))
  col_docs = nltk.TextCollection(docs)

  # Vocab
  vocab = list(set(col_query) | set(col_docs))
  vocab = map(lambda x: x.lower(), vocab)
  vocab = [w for w in vocab if not w in stopwords.words('english')]

  # Prep Vectors
  results = []
  for i in range(0, len(docs)):
    u = []
    v = []
    fd_doc_current = nltk.FreqDist(docs[i])
    temp_query = map(lambda x: x.lower(), query)
    temp_query = [w for w in temp_query if not w in stopwords.words('english')]
    temp_doc = map(lambda x: x.lower(), docs[i])
    temp_doc = [w for w in temp_doc if not w in stopwords.words('english')]
    for term in vocab:
      if term in temp_query:
        u.append(col_docs.tf_idf(term, temp_doc))
      else:
        u.append(0)
      if term in temp_doc:
        v.append(col_docs.tf_idf(term, temp_doc))
      else:
        v.append(0)
    if math.sqrt(numpy.dot(u, u)) == 0.0:
      results.append(1000)
    else:
      r = nltk.cluster.util.cosine_distance(u,v)
      results.append(r)
  return results.index(min(results))


def citProv(cite_key, context):
  # Use Citdensity to guess type
  citType = citDensity(context)
  if citType == 0:
    # General
    print citationTypes[citType]
  elif citType == 1:
    # Specific
    resultIndex = cosineSimilarity(cite_key,context1)
    print display_query
    print
    print docs[resultIndex]
  else:
    # Undetermined
    print citationTypes[citType]

citProv(cite_key,context1)












