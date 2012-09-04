execfile('Utils.py')
import Utils

p = Utils.pickler()
d = Utils.dataset_tools(Utils.dist(), Utils.tools())

raw = d.prepRaw(d.dist, d.tools, p.authors, p.experiment, p.titles)
print len(raw.keys())
p.dumpPickle(raw, "Raw")
