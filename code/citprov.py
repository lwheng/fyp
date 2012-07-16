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

CHUNK_SIZE = 10
citationTypes = ['General', 'Specific', 'Undetermined']

cite_key = "P03-2013==>P98-1085"
context1 = """<context citStr='Breiman et al. 1984' endWordPosition='1944' position='12827' startWordPosition='1941'>ntil the entire file has been processed for that one transformation, then regardless of the order of processing the output will be: ABBBBB, since the triggering environment of a transformation is always checked before that transformation is applied to any surrounding objects in the corpus. If the effect of a transformation is recorded immediately, then processing the string left to right would result in: ABABAB, whereas processing right to left would result in: ABBBBB. 3. A Comparison With Decision Trees The technique employed by the learner is somewhat similar to that used in decision trees (Breiman et al. 1984). A decision tree is trained on a set of preclassified entities and outputs a set of questions that can be asked about an entity to determine its proper classification. Decision trees are built by finding the question whose resulting partition is the purest,2 splitting the training data according to that question, and then recursively reapplying this procedure on each resulting subset. We first show that the set of classifications that can be provided via decision trees is a proper subset of those that can be provided via transformation lists (an ordered</context>"""

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
  col_query = nltk.TextCollection(query)


  # Cited Paper
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]
  citedpaper = "/Users/lwheng/Downloads/fyp/pdfbox-0.72/" + cited[0] + "/" + cited[0:3] + "/" + cited + ".txt"
  domain = []
  try:
    openfile = open(citedpaper,"r")
    for l in openfile:
      domain.append(nltk.word_tokenize(l.strip()))
    openfile.close()
  except IOError as e:
    print e
  docs = []
  for i in xrange(0, len(domain), CHUNK_SIZE/2):
    sublist = domain[i:i+CHUNK_SIZE]
    temp = []
    for s in sublist:
      temp.extend(s)
    docs.append(temp)
  col_docs = nltk.TextCollection(docs)

  # Vocab
  vocab = list(set(query) | set(col_docs))

  # Prep Vectors
  results = []
  for i in range(0, len(docs)):
    u = []
    v = []
    fd_doc_current = nltk.FreqDist(docs[i])
    for term in vocab:
      if term in query:
        u.append(col_query.tf_idf(term,docs[i]))
      else:
        u.append(0)
      if term in docs[i]:
        v.append(col_docs.tf_idf(term, docs[i]))
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
    print citationTypes[citType]
    resultIndex = cosineSimilarity(cite_key,context1)
    print display_query
    print
    print docs[resultIndex]
    print
    print list(set(query) & set(docs[3]))
  else:
    # Undetermined
    print citationTypes[citType]

citProv(cite_key,context1)












