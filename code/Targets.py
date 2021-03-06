import cPickle as pickle
import Utils
import os
import re
import numpy as np

if __name__ == '__main__':
  # Output
  y = []
  y_info = []
  y_hash_1st_tier = {}
  y_hash_2nd_tier = {}

  # Init
  general = 0
  spec_yes = 1
  spec_no = 2
  undetermined = 3

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
      context_id = int(obj[0][1])
      body_text_id = int(obj[0][2])
      annotation = obj[0][3]
      y_info.append((cite_key, context_id, body_text_id, annotation))
      if cite_key in y_hash_1st_tier:
        if context_id not in y_hash_1st_tier[cite_key]:
          y_hash_1st_tier[cite_key][context_id] = annotation
      else:
        temphash = {}
        temphash[context_id] = annotation
        y_hash_1st_tier[cite_key] = temphash

      if annotation == 'g':
        y.append(general)
      elif annotation == 'y':
        y.append(spec_yes)
        if cite_key not in y_hash_2nd_tier:
          context_hash = {}
          y_hash_2nd_tier[cite_key] = context_hash
        context_hash = y_hash_2nd_tier[cite_key]
        if context_id not in context_hash:
          body_text_hash = {}
          context_hash[context_id] = body_text_hash
        body_text_hash = context_hash[context_id]
        body_text_hash[body_text_id] = annotation
        context_hash[context_id] = body_text_hash
        y_hash_2nd_tier[cite_key] = context_hash
      elif annotation == 'n':
        y.append(spec_no)
      elif annotation == 'u':
        y.append(undetermined)

  y = np.asarray(y)
  print "Stats for y:"
  total_records = y.shape[0]
  print "Total number of records - " + str(total_records)
  general_records = float(y[y==0].shape[0])/(total_records)
  specific_yes_records = float(y[y==1].shape[0])/(total_records)
  specific_no_records = float(y[y==2].shape[0])/(total_records)
  undetermined_records = float(y[y==3].shape[0])/(total_records)
  print "General - " + str(general_records*100) + "%"
  print "Specific-Yes - " + str(specific_yes_records*100) + "%"
  print "Specific-No - " + str(specific_no_records*100) + "%"
  print "Undetermined - " + str(undetermined_records*100) + "%"
  pickle.dump(y, open(os.path.join(path_pickles,'Y.pickle'),'wb'))
  pickle.dump(y_info, open(os.path.join(path_pickles,'Y_Info.pickle'),'wb'))
  pickle.dump(y_hash_1st_tier, open(os.path.join(path_pickles,'Y_Hash_1st_Tier.pickle'),'wb'))
  pickle.dump(y_hash_2nd_tier, open(os.path.join(path_pickles,'Y_Hash_2nd_Tier.pickle'),'wb'))
