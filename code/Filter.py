import cPickle as pickle
import os

def checkQualify(citing, cited, path_parscit, path_parscit_section):
  # Check whether both citing and cited have the 2 types of file
  path_parscit_citing = os.path.join(path_parscit, citing + "-parscit.xml")
  path_parscit_section_citing = os.path.join(path_parscit_section, citing + "-parscit-section.xml")
  path_parscit_cited = os.path.join(path_parscit, cited + "-parscit.xml")
  path_parscit_section_cited = os.path.join(path_parscit_section, cited + "-parscit-section.xml")

  paths = [path_parscit_citing, path_parscit_section_citing, path_parscit_cited, path_parscit_section_cited]
  for p in paths:
    if not os.path.exists(p):
      return False
  return True

if __name__ == '__main__':
  # Output
  filtered = []

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']

  # Open Annotations Master
  annotationsMasterFile = "./annotationsMaster.txt"
  for l in open(annotationsMasterFile,'r'):
    entry = l.strip().replace(",","")
    entry_split = entry.split("==>")
    citing = entry_split[0]
    cited = entry_split[1]

    if checkQualify(citing, cited, path_parscit, path_parscit_section):
      f = {}
      f['citing'] = citing
      f['cited'] = cited
      filtered.append(f)

  # Dump pickle
  pickle.dump(filtered, open('Filtered.pickle','wb'))
