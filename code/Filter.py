import cPickle as pickle
import os
import re

def check_qualify(citing, cited, path_parscit, path_parscit_section):
  # Check whether both citing and cited have the 2 types of file
  path_parscit_citing = os.path.join(path_parscit, citing + "-parscit.xml")
  path_parscit_section_citing = os.path.join(path_parscit_section, citing + "-parscit-section.xml")
  path_parscit_cited = os.path.join(path_parscit, cited + "-parscit.xml")
  path_parscit_section_cited = os.path.join(path_parscit_section, cited + "-parscit-section.xml")

  paths = [path_parscit_citing, path_parscit_section_citing, path_parscit_cited, path_parscit_section_cited]
  for p in paths:
    if not os.path.exists(p):
      return False
    if not (os.path.getsize(p) > 19000):
      return False
  return True

if __name__ == '__main__':
  # Config
  num_of_records = 300 # Number of citation keys. Note that one citation key can have multiple contexts

  # Output
  filtered = []

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Open Annotations Master, and add all qualified
  qualified = []
  file_annotations_master = "./annotationsMaster.txt"
  for l in open(file_annotations_master,'r'):
    entry = l.strip().replace(",","")
    entry_split = entry.split("==>")
    citing = entry_split[0]
    cited = entry_split[1]

    if check_qualify(citing, cited, path_parscit, path_parscit_section):
      q = {}
      q['citing'] = citing
      q['cited'] = cited
      qualified.append(q)
  
  # First add all already annotated
  filtered = []
  file_annotations = "./Annotations.txt"
  regex = r"\#\d{3}\s+(.*)==>(.*),(.*)"
  for l in open(file_annotations, 'r'):
    obj = re.findall(regex, l.strip())
    cite = {}
    cite['citing'] = obj[0][0]
    cite['cited'] = obj[0][1]
    annotation = obj[0][2]
    if annotation and (cite in qualified):
      if cite not in filtered:
        filtered.append(cite)
  num_already_annotated = len(filtered)

  # Fill to num_of_records
  num_to_fill = num_of_records - num_already_annotated
  for q in qualified:
    if num_to_fill==0:
      break
    if q not in filtered:
      filtered.append(q)
      num_to_fill -= 1

  # Dump pickle
  pickle.dump(filtered, open(os.path.join(path_pickles,'Filtered.pickle'),'wb'))
