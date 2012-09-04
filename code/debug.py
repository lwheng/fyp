execfile('Utils.py')
import Utils
p = Utils.pickler()
d = Utils.dist()
t = Utils.tools()
dt = Utils.dataset_tools(d, t)
c = {'citing':'A00-1028', 'cited':'P94-1026'}
print dt.prepContexts(d, t, p.titles, c).toxml()
