#!/opt/local/bin/python
# -*- coding: utf-8 -*-

# To determine the type of the citation.
# If specific, we wish to determine the location of the cited information

# Feature Set
# 1. Citation Density
# 2. Publishing Year Difference
# 3. Title Overlap
# 4. Authors Overlap
# 5. Context's Average TF-IDF Weight
# 6. Location Of Citing Sentence?
# 7. Cosine Similarity
# 7.1 Cited Chunk's Average TF-IDF Weight

# Other possible features
# x+1. Cue Words 
# x+2. POS Tagging

from xml.dom.minidom import parseString
import unicodedata
import nltk
# import HTMLParser
import sys
import re
import os
import math
import numpy
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.metrics import distance

sentenceTokenizer = PunktSentenceTokenizer()

CHUNK_SIZE = 15
LAMBDA_AUTHOR_MATCH = 0.8
citationTypes = ['General', 'Specific', 'Undetermined']
genericHeader = [ 'abstract',
                  'acknowledgements',
                  'background',
                  'categories and subject descriptors',
                  'conclusions',
                  'discussions',
                  'evaluation',
                  'general terms',
                  'introduction',
                  'keywords',
                  'method',
                  'references',
                  'related work'
                ]

punctuation = "~`!@#$%^&*()-_+={}[]|\\:;\"\'<>,.?/"

titles = {}
authors = {}
# contextCollection = ""
collection = ""

# Pickle files
pickle_paperTitles = "/Users/lwheng/Downloads/fyp/paperTitles.pickle"
pickle_paperAuthors = "/Users/lwheng/Downloads/fyp/paperAuthors.pickle"
# pickle_contextCollection = "/Users/lwheng/Downloads/fyp/contextCollection.pickle"

# def fetchContextCollection():
#   print "Loading context collection..."
#   tempCol = pickle.load(open(pickle_contextCollection,"rb"))
#   print "Loaded context collection"
#   return tempCol

def fetchTitles():
  tempTitle = {}
  tempTitle = pickle.load(open(pickle_paperTitles, "rb"))
  # print "Loaded titles"
  return tempTitle

def fetchAuthors():
  tempAuthors = {}
  tempAuthors = pickle.load(open(pickle_paperAuthors, "rb"))
  # print "Loaded authors"
  return tempAuthors

def fetchContexts(cite_key):
  global titles
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]

  titleToMatch = titles[cited]

  citingFile = "/Users/lwheng/Downloads/fyp/parscitxml500/" + citing + "-parscit.xml"
  openciting = open(citingFile,"r")
  data = openciting.read()
  openciting.close()
  dom = parseString(data)
  citations = dom.getElementsByTagName('citation')
  tags = ["title", "note", "booktitle", "journal"]
  titleTag = []
  index = 0
  bestIndex = 0
  minDistance = 314159265358979323846264338327950288419716939937510
  for i in range(len(citations)):
    c = citations[i]
    valid = c.getAttribute('valid')
    if valid == "true":
      titleTag = []
      index = 0
      while titleTag == []:
        titleTag = c.getElementsByTagName(tags[index])
        index += 1
      title = titleTag[0].firstChild.data
      title = unicodedata.normalize('NFKD', title).encode('ascii','ignore')
      thisDistance = levenshtein(title, titleToMatch)
      if thisDistance < minDistance:
        minDistance = thisDistance
        bestIndex = i
  return citations[bestIndex]

def levenshtein(a, b):
  return distance.edit_distance(a,b)

def levenshteinRatio(a, b):
  lensum = float(len(a) + len(b))
  if lensum == 0.0:
    return 1.0
  ldist = levenshtein(a,b)
  return (lensum - ldist) / lensum

def jaccard(inputA, inputB):
  # Returns jaccard index. Smaller the more better
  a = inputA.lower()
  b = inputB.lower()
  return distance.jaccard_distance(set(a.split()), set(b.split()))

def citDensity(context_lines, context_citStr):
  # Regular Expression
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
  # regex = r"(((\w+)\s*,?\s*(et al.?)?|(\w+ and \w+))\s*,?\s*(\(?\s?\d{4}\s?\)?)|\[\s*(\w+)\s*\]|\[\s(\w+\d+)\s\]|[\[|\(]\s(\d+\s?,\s?)*(\d+)\s[\]|\)]|\(\s*[A-Z]\w+\s*\)|\[\s(\w+\s,?\s?)+\])"
  output = []
  # Process citStr
  if "et al." in context_citStr:
    context_citStr = context_citStr.replace("et al.", "et al")
  # Process context
  if "et al." in context_lines:
    context_lines = context_lines.replace("et al.", "et al")
  query_lines = sentenceTokenizer.tokenize(context_lines)
  # for l in query_lines:
  #   obj = re.findall(regex, l)
  #   if context_citStr in l:
  #     output.append((True, len(obj)))
  #   else:
  #     output.append((False, len(obj)))
  # density = 0
  # avgdensity = 0
  # for t in output:
  #   if t[0]:
  #     density += t[1]
  #   else:
  #     avgdensity += t[1]
  citationCount = 0
  for l in query_lines:
    obj = re.findall(regex, l)
    citationCount += len(obj)
  avgDensity = float(citationCount) / float(len(query_lines))
  return avgDensity

def publishYear(cite_key, context_citStr):
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]

  citingYear = int(citing[1:3])
  citedYear = int(cited[1:3])

  if citingYear > 50:
    citingYear = 1900 + citingYear
  else:
    citingYear = 2000 + citingYear

  if citedYear > 50:
    citedYear = 1900 + citedYear
  else:
    citedYear = 2000 + citedYear
  return (citingYear-citedYear)

def titleOverlap(cite_key):
  global titles
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]
  return jaccard(titles[citing], titles[cited])

def authorOverlap(cite_key):
  global authors
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]
  # Adapting the Jaccard idea
  matches = 0
  uniqueNames = len(authors[citing]) + len(authors[cited])
  for citingAuthor in authors[citing]:
    for citedAuthor in authors[cited]:
      ratio = levenshteinRatio(citingAuthor, citedAuthor)
      if ratio > LAMBDA_AUTHOR_MATCH:
        matches += 1
        uniqueNames -= 1
  if uniqueNames == 0:
    return 1.0
  return float(matches) / float(uniqueNames)

def chunkAverageWeight(chunk, collection):
  weight = 0
  for t in chunk.tokens:
    weight += collection.tf_idf(t.lower(), chunk)
  return float(weight) / float(len(chunk.tokens))

def locationCitingSent(cite_key):
  info = cite_key.split("==>")
  citing = info[0]

  parscitSectionFile = "/Users/lwheng/Downloads/fyp/parscitsectionxml500/" + citing + "-parscit-section.xml"
  if os.path.exists(parscitSectionFile):
    print
  openthefile = open(parscitSectionFile, 'r')
  data = openthefile.read()
  openthefile.close()
  dom = parseString(data)

def cosineSimilarity(cite_key, query_tokens, query_col):
  global CHUNK_SIZE

  # # Citing Paper
  # query_col = nltk.TextCollection([nltk.Text(query_tokens)])

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
  docs_display = []
  for i in xrange(0, len(domain), CHUNK_SIZE/2):
    sublist = domain[i:i+CHUNK_SIZE]
    temp = ""
    for s in sublist:
      temp = temp + " " + s
    text = nltk.Text(nltk.word_tokenize(temp.lower()))
    docs.append(text)
    docs_display.append((str(i) + "-" + str(i+CHUNK_SIZE), text))
  docs_col = nltk.TextCollection(docs)

  # Vocab
  vocab = list(set(query_col) | set(docs_col))
  vocab = map(lambda x: x.lower(), vocab)
  vocab = [w for w in vocab if not w in stopwords.words('english')]
  vocab = [w for w in vocab if not w in punctuation]

  # Prep Vectors
  results = []
  for i in range(0, len(docs)):
    # 7.1 Cited Chunk's Average TF-IDF Weight
    chunkAvgWeight = chunkAverageWeight(docs[i], docs_col)

    u = []
    v = []
    # fd_doc_current = nltk.FreqDist(docs[i])
    temp_query = map(lambda x: x.lower(), query_tokens)
    temp_query = [w for w in temp_query if not w in stopwords.words('english')]
    temp_query = [w for w in temp_query if not w in punctuation]
    temp_doc = map(lambda x: x.lower(), docs[i])
    temp_doc = [w for w in temp_doc if not w in stopwords.words('english')]
    temp_doc = [w for w in temp_doc if not w in punctuation]
    for term in vocab:
      if term in temp_query:
        u.append(docs_col.tf_idf(term, temp_doc))
      else:
        u.append(0)
      if term in temp_doc:
        v.append(docs_col.tf_idf(term, temp_doc))
      else:
        v.append(0)
    if math.sqrt(numpy.dot(u, u)) == 0.0:
      results.append(0.0)
    else:
      r = nltk.cluster.util.cosine_distance(u,v)
      # results.append(r)
      results.append((r,chunkAvgWeight))
  # total = sum(results)
  feature = []
  for i in range(len(results)):
    feature.append((docs_display[i][0],results[i]))
    # feature.append((docs_display[i][0],float(results[i])/float(total)*100))
  return feature

def citProv(cite_key):
  info = cite_key.split('==>')
  citing = info[0]
  cited = info[1]

  citation_dom = fetchContexts(cite_key)
  contexts = citation_dom.getElementsByTagName('context')

  # Prep citing_col
  context_list = []
  for c in contexts:
    value = c.firstChild.data
    value = unicodedata.normalize('NFKD', value).encode('ascii','ignore')
    value = value.lower()
    tempText = nltk.Text(nltk.word_tokenize(value))
    context_list.append(tempText)
  citing_col = nltk.TextCollection(context_list)

  for c in contexts:
    feature_vector = []

    context_citStr = c.getAttribute('citStr')
    context_citStr = unicodedata.normalize('NFKD', context_citStr).encode('ascii','ignore')
    context_value = c.firstChild.data
    context_value = unicodedata.normalize('NFKD', context_value).encode('ascii','ignore')

    query_lines = context_value
    query_tokens = nltk.word_tokenize(context_value.lower())
    query_col = nltk.TextCollection([nltk.Text(query_tokens)])
    query_display = ""
    for t in query_tokens:
      query_display = query_display + " " + t

    # 1. Using Citation Density
    feature_citDensity = citDensity(query_lines, context_citStr)
    feature_vector.append(feature_citDensity)   

    # 2. Publishing Year Difference
    feature_publishYear = publishYear(cite_key, context_citStr)
    feature_vector.append(feature_publishYear)

    # 3. Title Overlap
    feature_titleOverlap = titleOverlap(cite_key)
    feature_vector.append(feature_titleOverlap)

    # 4. Authors Overlap
    feature_authorOverlap = authorOverlap(cite_key)
    feature_vector.append(feature_authorOverlap)

    # 5. Context's Average TF-IDF Weight
    feature_queryWeight = chunkAverageWeight(nltk.Text(query_tokens), citing_col)
    feature_vector.append(feature_queryWeight)

    # 6. Location Of Citing Sentence
    # feature_locationCitingSent = locationCitingSent(cite_key)
    # feature_vector.append(feature_locationCitingSent)

    # 7. Cosine Similarity + 7.1 Cited Chunk's Average TF-IDF Weight
    # Note: For n chunks in cited paper we perform cosineSimilarity,
    # so we have n results
    feature_cosineSimilarity = cosineSimilarity(cite_key, query_tokens, query_col)
    feature_vector.append(feature_cosineSimilarity)

    # print cite_key + " : " + str(feature_vector[0:-1])
    for i in feature_cosineSimilarity:
      display = feature_vector[0:-1]
      display.append(i[1])
      print cite_key + " : " + str(i[0]) + " : " + str(display)
      # print i[0] + "\t" + str(i[1])

experiment50 = "/Users/lwheng/Dropbox/fyp/annotation/annotations50.txt"
startexperiment = open(experiment50,"r")
titles = fetchTitles()
authors = fetchAuthors()
# contextCollection = fetchContextCollection()
for l in startexperiment:
  info = l.split(",")
  cite_key = info[0]
  citProv(cite_key)
  sys.exit()












