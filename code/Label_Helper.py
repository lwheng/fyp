from xml.dom.minidom import parseString
from flask import Flask
import cPickle as pickle
import os

app = Flask(__name__)

# Load Config.pickle
config = pickle.load(open('Config.pickle','r'))
path_parscit = config['path_parscit']
path_parscit_section = config['path_parscit_section']
path_pickles = config['path_pickles']

# Load Contexts
#contexts = pickle.load(open(os.path.join(path_pickles,'Contexts.pickle'),'r'))
# Load Filtered
#filtered = pickle.load(open(os.path.join(path_pickles,'Filtered.pickle'),'r'))
# Load For_Labelling
#for_labelling = pickle.load(open(os.path.join(path_pickles,'Filtered.pickle'),'r'))

def printer(self, citing, cited, citStr, wholeText, citedLines):
  display = "<div style='width:100%'>"
  display += "<h2>" + citing + " cites " + cited + "</h2>"
  display += "<a href='http://aclweb.org/anthology/" + citing[0] + "/" + citing[0:3] + "/" + citing + ".pdf'>citing</a>"
  display += "<p>"
  display += "<a href='http://aclweb.org/anthology/" + cited[0] + "/" + cited[0:3] + "/" + cited+ ".pdf'>cited</a>"
  display += "<h2>citStr = " + citStr + "</h2>"
  display += "<div style='float:left; display:inline-block; width:40%; overflow:auto' class='div1'>"
  display += wholeText
  display += "</div>"
  display += "<div style='float:right; display:inline-block; width:50%; height:400px; overflow:auto' class='div2'>"
  for i in range(len(citedLines)):
    l = citedLines[i]
    display += "<div>" + str(i+1) + ". " + l + "</div>"
  display += "</div>"
  display += "</div>"
  return display
    

@app.route("/")
def hello():
  return "<h1>Hello, Low Wee. Welcome to annotating.</h1>"

@app.route("/<int:item_id>/<int:context_id>")
def show(item_id, context_id):
  item = for_labelling[item_id]
  hash_key = item[0]
  cited = hash_key.split("==>")[1]
  this_contexts = contexts[hash_key]
  c = this_contexts[context_id]

  data = open(os.path.join(path_parscit_section, 'cited'+'-parscit-section.xml'),'r').read()
  dom = parseString(data)
  bodyTexts = dom.getElementsByTagName('bodyText')
  return printer(hash_key, c, bodyTexts)

if __name__ == '__main__':
  app.run() 
