import cPickle as pickle
import os

if __name__ == '__main__':
  # Output
  lines = []

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load For_Labelling.pickle
  #for_labelling = pickle.load(open(os.path.join(path_pickles, 'For_Labelling.pickle'), 'r'))
  for_labelling = pickle.load(open(os.path.join(path_pickles, 'For_Labelling_Plus.pickle'), 'r'))

  for i in range(len(for_labelling)):
    f = for_labelling[i]
    cite_key = f[0]
    c_id = f[1]
    b_id = f[2]
    print cite_key + "  " + str(c_id) + "  " + str(b_id).rjust(3, '0') + "  " + ":"
