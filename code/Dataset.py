import sys
import os
import Utils

class dataset:
  def __init__(self, rootDirectory="/Users/lwheng/Downloads/fyp/"):
    self.parscitSectionPath = os.path.join(rootDirectory, "parscitsectionxml")
    self.parscitPath = os.path.join(rootDirectory, "parscitxml")
    self.tools = Utils.tools()
    self.dist = Utils.dist()
    self.titles = Utils.pickler().titles
    self.authors = Utils.pickler().authors

  def fetchExperiment(self, experimentFile="/Users/lwheng/Dropbox/fyp/annotation/annotations500.txt"):
    openfile = open(experimentFile,'r')
    experiment = []
    for l in openfile:
      info = l.strip().split('==>')
      data = {}
      data['citing'] = info[0]
      data['cited'] = info[1]
      experiment.append(data)
    return experiment

  def fetchContexts(self, cite_key, titles):
    citing = cite_key['citing']
    cited = cite_key['cited']
    titleToMatch = titles[cited]

    citingFile = os.path.join(self.parscitPath, citing+"-parscit.xml")
    openciting = open(citingFile,"r")
    data = openciting.read()
    openciting.close()
    dom = self.tools.parseXML(data)
    citations = dom.getElementsByTagName('citation')
    tags = ["title", "note", "booktitle", "journal", "tech", "author"]
    titleTag = []
    index = 0
    bestIndex = -1
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
        title = self.tools.normalize(title)
        thisDistance = self.dist.levenshtein(title, titleToMatch)
        if thisDistance < minDistance:
          minDistance = thisDistance
          bestIndex = i
    if bestIndex == -1:
      return None
    return citations[bestIndex]

  def fetchDataset(self):
    experiments = self.fetchExperiment()
    for e in experiments:
      print self.fetchContexts(e, self.titles).toxml()
