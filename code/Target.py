execfile('Utils.py')
import Utils

if __name__ == "__main__":
  pickler = Utils.pickler()
  pathAnnotations = "/Users/lwheng/Dropbox/fyp/code/Annotations.txt"
  dataset_tools = Utils.dataset_tools(Utils.dist(), Utils.tools())
  target = dataset_tools.prepTarget(pathAnnotations)
  print len(target)
  pickler.dumpPickle(target, "Target")
