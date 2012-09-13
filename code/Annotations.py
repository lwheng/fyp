execfile('Utils.py')
import Utils

if __name__ == "__main__":
  pickler = Utils.pickler()
  pathAnnotations = "/Users/lwheng/Dropbox/fyp/code/Annotations.txt"
  dataset_tools = Utils.dataset_tools(Utils.dist(), Utils.nltk_tools(), Utils.tools())
  y = dataset_tools.prepAnnotations(pathAnnotations)
  pickler.dumpPickle(y, "Annotations")
