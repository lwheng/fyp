import cPickle as pickle
import os

if __name__ == "__main__":
  # Output
  for_labelling_file = []
  for_labelling = []

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load Filtered pickle
  filtered = pickle.load(open(os.path.join(path_pickles,'Filtered.pickle'),'r'))

  # Load Doms
  doms = pickle.load(open(os.path.join(path_pickles,'Doms.pickle'),'r'))

  # Load Contexts
  contexts = pickle.load(open(os.path.join(path_pickles,'Contexts.pickle'),'r'))

  num = len(filtered)
  for f in filtered:
    citing = f['citing']
    cited = f['cited']
    hash_key = citing + "==>" + cited

    # Get context
    this_context = contexts[hash_key]

    # Get dom tuple
    dom_tuple = doms[hash_key]
    dom_parscit_section_cited = dom_tuple[3]
    #body_texts = dom_parscit_section_cited.getElementsByTagName('bodyText')
    body_texts = dom_parscit_section_cited.getElementsByTagName('variant')[0].childNodes

    index = 0
    for c in range(len(this_context)):
      bindex = 0
      for b in range(len(body_texts)):
        temp = body_texts[b]
        if temp.nodeType == temp.TEXT_NODE:
          continue
        line = str(index).rjust(3,'0') + "\t" + hash_key + "\t" + "c"+str(c) + "\t" + "bt"+str(b) + "\t:"
        for_labelling_file.append(line)
        #for_labelling.append((hash_key, c, b))
        for_labelling.append((hash_key, c, bindex))
        index += 1
        bindex += 1
    num -= 1
    print "No. left = " + str(num)

  # Dump pickle
  pickle.dump(for_labelling, open(os.path.join(path_pickles, "For_Labelling.pickle"),'wb'))
