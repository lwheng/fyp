import cPickle as pickle
import Utils
import os
import re
import numpy as np

if __name__ == '__main__':
  y = []

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load and open Labels.txt
  labels_file = open('Labels.txt','r')

  # Parse Labels.txt
  regex = r"(.*)\s+(\d{1})\s+(\d{3})\s+\:(\w)"
  for l in labels_file:
    obj = re.findall(regex,l.strip())
    if obj:
      cite_key = obj[0][0].strip()
      context_id = obj[0][1]
      body_text_id = int(obj[0][2])
      annotation = obj[0][3]
      if annotation == 'g':
        y.append(0)
      elif annotation == 'y':
        y.append(1)
      elif annotation == 'n':
        y.append(2)
      elif annotation == 'u':
        y.append(3)

  y = np.asarray(y)
  pickle.dump(y, open(os.path.join(path_pickles,'Y.pickle'),'wb'))
