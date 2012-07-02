#!/usr/bin/python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parseString
import unicodedata
import nltk
import HTMLParser
import sys
h = HTMLParser.HTMLParser()

contexts = []
contexts.append("""<context position="3648" citStr="Caraballo (1999)" startWordPosition="592" endWordPosition="593">. In Section 4, we show how correctly extracted relationships can be used as “seed-cases” to extract several more relationships, thus improving recall; this work shares some similarities with that of Caraballo (1999). In Section 5 we show that combining the techniques of Section 3 and Section 4 improves both precision and recall. Section 6 demonstrates that 1Another possible view is that “hyponymy” should only re</context>""")
contexts.append("""<context position="4278" citStr="Caraballo (1999)" startWordPosition="695" endWordPosition="696">ingent relationships are an important part of world-knowledge (and are therefore worth learning), and because in practice we found the distinction difficult to enforce. Another definition is given by Caraballo (1999): “... a word A is said to be a hypernym of a word B if native speakers of English accept the sentence ‘B is a (kind of) A.’ ” linguistic tools such as lemmatization can be used to reliably put the ex</context>""")
contexts.append("""<context position="17183" citStr="Caraballo (1999)" startWordPosition="2851" endWordPosition="2852">hat might be found in text are expressed overtly by the simple lexicosyntactic patterns used in Section 2, as was apparent in the results presented in that section. This problem has been addressed by Caraballo (1999), who describes a system that first builds an unlabelled hierarchy of noun clusters using agglomerative bottom-up clustering of vectors of noun coordination information. The leaves of this hierarchy (</context>""")
contexts.append("""<context position="17183" citStr="Caraballo (1999)" startWordPosition="2851" endWordPosition="2852">the big brown fox jumps over the lazy dog</context>""")
t = []
for i in range(0, len(contexts)):
	context = contexts[i]
	domCiting = parseString(context)
	nodeContext = domCiting.getElementsByTagName("context")
	citStr = nodeContext[0].attributes['citStr'].value
	contextData = nodeContext[0].firstChild.data
	contextDataString = unicodedata.normalize('NFKD', contextData).encode('ascii','ignore').replace(citStr, "")
	text = nltk.word_tokenize(contextDataString)
	t.append(nltk.Text(text))
	
col = nltk.TextCollection(t)
print t[0].count('Section')
print col.tf('Section', t[0])
print col.idf('Section')
print col.tf_idf('Section',t[0])
# tagging = nltk.pos_tag(text)

citedpaper = "/Users/lwheng/Desktop/P99-1016-parscit-section.xml"
try:
	opencited = open(citedpaper,"r")
	data = opencited.read()
	dom = parseString(data)
	title = dom.getElementsByTagName("title")
	bodyText = dom.getElementsByTagName("bodyText")
	# for i in bodyText:
		# print h.unescape(i.firstChild.data)
		# print i.firstChild.data
except IOError as e:
  print "Error"
