#!/opt/local/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import re

# Compiles all the paper titles for easy reference

metadataDir = "/Users/lwheng/Downloads/fyp/metadata/"
files = os.listdir(metadataDir)
for f in files:
	filename = metadataDir + f
	openfile = open(filename,"r")
	reOpenPaper = r"<paper id=\"(.*)\">"
	reClosePaper = r"</paper>"
	reTitle = r"<title>(.*)</title>"
	opentitle = False
	for l in openfile:
		matchOpen = re.findall(reOpenPaper,l.strip())
		if matchOpen:
			opentitle = True
			paperid = matchOpen[0]
		if opentitle:
			matchTitle = re.findall(reTitle,l.strip())
			if matchTitle:
				print f[0:3] + "-" + paperid + "==>" + matchTitle[0]
				opentitle = False
	openfile.close()