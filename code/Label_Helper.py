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
contexts = pickle.load(open(os.path.join(path_pickles,'Contexts.pickle'),'r'))
# Load Doms
#doms = pickle.load(open(os.path.join(path_pickles,'Doms.pickle'),'r'))
# Load Filtered
filtered = pickle.load(open(os.path.join(path_pickles,'Filtered.pickle'),'r'))
# Load For_Labelling
for_labelling = pickle.load(open(os.path.join(path_pickles,'For_Labelling.pickle'),'r'))

def printer(citing, cited, num_contexts, cit_str, context, body_texts):
  display = "<div style='width:100%'>"
  display += "<h2>" + citing + " cites " + cited + "</h2>"
  display += "<a href='http://aclweb.org/anthology/" + citing[0] + "/" + citing[0:3] + "/" + citing + ".pdf'>citing</a>"
  display += "<p>"
  display += "<a href='http://aclweb.org/anthology/" + cited[0] + "/" + cited[0:3] + "/" + cited+ ".pdf'>cited</a>"
  display += "<h2>citStr = " + cit_str + "</h2>"
  display += "<h3>No. contexts = " + str(num_contexts) + ", No. bodyText = " + str(len(body_texts)) + "</h3>"
  display += "<div style='float:left; display:inline-block; width:40%; overflow:auto' class='div1'>"
  display += context
  display += "</div>"
  display += "<div style='float:right; display:inline-block; width:50%; height:400px; overflow:auto' class='div2'>"
  for i in range(len(body_texts)):
    b = body_texts[i]
    whole_text = b.firstChild.wholeText
    display += "<div>" + "<strong>Body Text Index = " + str(i) + "</strong></div>"
    display += "<div>" + whole_text + "</div>"
    display += "<p>"
    display += "<p>"
  display += "</div>"
  display += "</div>"
  return display
    

@app.route("/")
def hello():
  return "<h1>Hello, Low Wee. Welcome to annotating.</h1>"

@app.route("/<int:item_id>/<int:context_id>")
def show(item_id, context_id):
  t = for_labelling[item_id]
  cite_key = t[0]
  citing = cite_key.split("==>")[0]
  cited = cite_key.split("==>")[1]

  c_list = contexts[cite_key]
  if c_list:
    # Has contexts
    c = c_list[context_id]
    num_contexts = len(c_list)
    cit_str = c.getAttribute('citStr')
    context = c.firstChild.wholeText

    # Get dom_parscit_section_cited
    path_parscit_section_cited = os.path.join(path_parscit_section, cited + "-parscit-section.xml")
    openfile = open(path_parscit_section_cited,'r')
    data = openfile.read()
    openfile.close()
    dom_parscit_section_cited = parseString(data)
    body_texts = dom_parscit_section_cited.getElementsByTagName('bodyText')
    return printer(citing, cited, num_contexts, cit_str, context, body_texts)
  else:
    # No contexts
    return "No Contexts"

if __name__ == '__main__':
  app.run() 
