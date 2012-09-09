from flask import Flask
import cPickle as pickle
import os
import unicodedata
app = Flask(__name__)

DatasetTBAFile = "/Users/lwheng/Downloads/fyp/For_Annotation.pickle"
DatasetTBA = pickle.load(open(DatasetTBAFile,'r'))

pdfboxPath = "/Users/lwheng/Downloads/fyp/pdfbox-0.72"

def normalize(text):
  return unicodedata.normalize('NFKD',text).encode('ascii', 'ignore')

@app.route("/")
def hello():
  return "<h1>Hello, Low Wee. Welcome to annotating.</h1>"

@app.route("/<int:context_id>")
def show(context_id):
  entry = DatasetTBA[context_id]
  info = entry[0]
  citing = info['citing']
  cited = info['cited']

  context_dom = entry[1]
  wholeText = normalize(context_dom.firstChild.wholeText)
  citStr = normalize(context_dom.attributes['citStr'].value)

  txtfile = os.path.join(pdfboxPath, cited[0], cited[0:3], cited+".txt")
  citedlines = []
  for l in open(txtfile,'r'):
    citedlines.append(l.strip())

  execfile('annotator_helper.py')
  from annotator_helper import annotator_helper
  helper = annotator_helper()
  return helper.printer(citing, cited, citStr, wholeText, citedlines)

if __name__ == '__main__':
  app.run()
