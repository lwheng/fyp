import sys
import cPickle as pickle

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print "USAGE: python Config.py [mac | ubuntu | wing]"
    sys.exit()
  rootDir = ""
  codeDir = ""
  option = sys.argv[1]
  if option == "mac":
    rootDir = "/Users/lwheng/Downloads/fyp"
    codeDir = "/Users/lwheng/Dropbox/fyp/code"
  elif option == "ubuntu":
    rootDir = "/home/lwheng/Desktop"
    codeDir = "/home/lwheng/Dropbox/fyp/code"
  elif option == "wing":
    rootDir = "/home/lwheng/fypsource"
    codeDir = "/home/lwheng/fyp/code"
  config = (rootDir, codeDir)
  pickle.dump(config, open("Config.pickle", "wb"))
