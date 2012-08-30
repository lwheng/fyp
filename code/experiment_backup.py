#!/usr/bin/python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parseString
import unicodedata
import nltk
import HTMLParser
import sys
import re
h = HTMLParser.HTMLParser()

# nltk.corpus.stopwords.words('english')
# nltk.cluster.util.cosine_distance(u,v)
# nltk.FreqDist(col).keys()

# citing paper
context = """<context position="5645" citStr="Mayfield et al., 2003" startWordPosition="809" endWordPosition="812">ning technique, which has been successfully applied to various natural language processing tasks including chunking tasks such as phrase chunking (Kudo and Matsumoto, 2001) and named entity chunking (Mayfield et al., 2003). In the preliminary experimental evaluation, we focus on 52 expressions that have balanced distribution of their usages in the newspaper text corpus and are among the most difficult ones in terms of [1]</context>"""
dom = parseString(context)
linesContext = dom.getElementsByTagName('context')[0].firstChild.data
linesContext = unicodedata.normalize('NFKD', linesContext).encode('ascii','ignore')
query = nltk.word_tokenize(linesContext)
query_display = ""
for i in query:
	query_display = query_display + " " + i
fd_query = nltk.FreqDist(query)

reg = []
reg.append(r"\(\s?(\d{1,3})\s?\)")
reg.append(r"\(\s?(\d{4})\s?\)")
reg.append(r"\(\s?(\d{4};?\s?)+\s?")
reg.append(r"\[\s?(\d{1,3},?\s?)+\s?\]")
reg.append(r"\[\s?([\w-],?\s?)+\s?\]")
reg.append(r"([A-Z][a-z-]+\s?,?\s?(\(and|&)\s)?)+\s?,?\s?(et al.)?\s?,?\s?(\(?(\d{4})\)?)")

regex = ""
for i in range(len(reg)):
	regex += reg[i] + "|"
regex = re.compile(regex[:-1])

# for citation density
# regex = r"((([A-Z][a-z]+)\s*(et al.?)?|([A-Z][a-z]+ and [A-Z][a-z]+))\s*,?\s*(\(?\d{4}\)?)|\[\s*(\d+)\s*\])"
obj = re.findall(regex, query_display)
print len(query)
print len(obj)
citation_density = float(len(obj)) / float(len(query))
print citation_density * 100
print obj
print

# cited paper
citedpapercode = "W03-0429"
citedpaper = "/Users/lwheng/Downloads/fyp/pdfbox-0.72/" + citedpapercode[0] + "/" + citedpapercode[0:3] + "/" + citedpapercode + ".txt"
t = []
SIZE = 10
lines = []
try:
	openfile = open(citedpaper,"r")
	for l in openfile:
		lines.append(nltk.word_tokenize(l.strip()))
	openfile.close()
except IOError as e:
	print e	
doc = []
for i in xrange(0, len(lines), SIZE):
	sublist = lines[i:i+SIZE]
	temp = []
	for s in sublist:
		temp.extend(s)
	doc.append(temp)

query_col = nltk.TextCollection(query)
col = nltk.TextCollection(doc)

# print "CITED"
# print col.collocations(50,3)
# print
# print col.common_contexts(["training"],20) # very similar to collocations
# print
# print col.concordance("training")
# print
# print col.generate() # Print random text, generated using a trigram language model.
# print
# print col.similar('training')
# print
# print 'End'

# prep vectors
vocab = list(set(query) | set(col))
u = []
v = []
results = []
for i in range(0,len(doc)):
	fd_doc0 = nltk.FreqDist(doc[i])
	for term in vocab:
		if term in query:
			# u.append(fd_query[term]) # using just frequency
			u.append(query_col.tf_idf(term, doc[i])) # using tf-idf weighting scheme
		else:
			u.append(0)
		if term in doc[i]:
			# v.append(fd_doc0[term]) # using just frequency
			v.append(col.tf_idf(term, doc[i])) # using tf-idf weighting scheme
		else:
			v.append(0)
	r = nltk.cluster.util.cosine_distance(u,v)
	results.append(r)
	
print "QUERY"
print query_display
print
toprint = ""
for i in doc[results.index(max(results))]:
	toprint = toprint + " " + i
print "GUESS"
print toprint
print
print max(results)
print results
	
	
	
	
		
# 
# contexts = []
# contexts.append("""<context position="3648" citStr="Caraballo (1999)" startWordPosition="592" endWordPosition="593">. In Section 4, we show how correctly extracted relationships can be used as “seed-cases” to extract several more relationships, thus improving recall; this work shares some similarities with that of Caraballo (1999). In Section 5 we show that combining the techniques of Section 3 and Section 4 improves both precision and recall. Section 6 demonstrates that 1Another possible view is that “hyponymy” should only re</context>""")
# contexts.append("""<context position="4278" citStr="Caraballo (1999)" startWordPosition="695" endWordPosition="696">ingent relationships are an important part of world-knowledge (and are therefore worth learning), and because in practice we found the distinction difficult to enforce. Another definition is given by Caraballo (1999): “... a word A is said to be a hypernym of a word B if native speakers of English accept the sentence ‘B is a (kind of) A.’ ” linguistic tools such as lemmatization can be used to reliably put the ex</context>""")
# contexts.append("""<context position="17183" citStr="Caraballo (1999)" startWordPosition="2851" endWordPosition="2852">hat might be found in text are expressed overtly by the simple lexicosyntactic patterns used in Section 2, as was apparent in the results presented in that section. This problem has been addressed by Caraballo (1999), who describes a system that first builds an unlabelled hierarchy of noun clusters using agglomerative bottom-up clustering of vectors of noun coordination information. The leaves of this hierarchy (</context>""")
# 
# t = []
# for i in range(0, len(contexts)):
#   context = contexts[i]
#   domCiting = parseString(context)
#   nodeContext = domCiting.getElementsByTagName("context")
#   citStr = nodeContext[0].attributes['citStr'].value
#   contextData = nodeContext[0].firstChild.data
#   contextDataString = unicodedata.normalize('NFKD', contextData).encode('ascii','ignore').replace(citStr, "")
#   text = nltk.word_tokenize(contextDataString)
#   t.append(nltk.Text(text))
#   
# col = nltk.TextCollection(t)
# print t[0].count('Section')
# print col.tf('Section', t[0])
# print col.idf('Section')
# print col.tf_idf('Section',t[0])
# # tagging = nltk.pos_tag(text)
# 
# citedpaper = "/Users/lwheng/Desktop/P99-1016-parscit-section.xml"
# try:
#   opencited = open(citedpaper,"r")
#   data = opencited.read()
#   dom = parseString(data)
#   title = dom.getElementsByTagName("title")
#   bodyText = dom.getElementsByTagName("bodyText")
#   # for i in bodyText:
#     # print h.unescape(i.firstChild.data)
#     # print i.firstChild.data
# except IOError as e:
#   print "Error"