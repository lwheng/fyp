import cPickle as pickle
import os
from xml.dom.minidom import parseString

if __name__ == "__main__":
  # Output
  doms = []

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']

  # Load Filtered pickle
  filtered = pickle.load(open(os.path.join(path_pickles,'Filtered.pickle'),'r'))

  # Open files and extract DOMs
  for f in filtered:
    print f
    citing = f['citing']
    cited = f['cited']

    # Open and parse citing
    path_parscit_citing = os.path.join(path_parscit, citing + "-parscit.xml")
    path_parscit_section_citing = os.path.join(path_parscit_section, citing + "-parscit-section.xml")
    openfile = open(path_parscit_citing,'r')
    data = openfile.read()
    openfile.close()
    dom_parscit_citing = parseString(data)
    openfile = open(path_parscit_citing,'r')
    data = openfile.read()
    openfile.close()
    dom_parscit_section_citing = parseString(data)
    
    # Open and parse cited
    path_parscit_cited = os.path.join(path_parscit, cited + "-parscit.xml")
    path_parscit_section_cited = os.path.join(path_parscit_section, cited + "-parscit-section.xml")
    openfile = open(path_parscit_cited,'r')
    data = openfile.read()
    openfile.close()
    dom_parscit_cited = parseString(data)
    openfile = open(path_parscit_cited,'r')
    data = openfile.read()
    openfile.close()
    dom_parscit_section_cited = parseString(data)

    # Append to output
    doms.append((dom_parscit_citing, dom_parscit_section_citing, dom_parscit_cited, dom_parscit_section_cited))

    # Dump pickle
    pickle.dump(doms, open('Doms.pickle','wb'))
