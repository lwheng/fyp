import sys
import cPickle as pickle

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print "USAGE: python Config.py [mac | citweb | wing]"
    sys.exit()
  option = sys.argv[1]
  config = {}
  if option == "mac":
    config['path_parscit'] = "/Users/lwheng/Downloads/fyp/parscitxml"
    config['path_parscit_section'] = "/Users/lwheng/Downloads/fyp/parscitsectionxml"
    config['path_pickles'] = "/Users/lwheng/Downloads/fyp/Pickles"
  elif option == "wing":
    config['path_parscit'] = "/home/lwheng/fypsource/parscitxml"
    config['path_parscit_section'] = "/home/lwheng/fypsource/parscitsectionxml"
    config['path_pickles'] = "/home/lwheng/fypsource/Pickles"
  elif option == "citweb":
    print
  pickle.dump(config, open("Config.pickle", "wb"))
