execfile('Utils.py')
import Utils
p = Utils.pickler()
d = Utils.dist()
t = Utils.tools()
dt = Utils.dataset_tools(d, t)
cite_key = 'P98-1117==>J93-1006'
c = {'citing':cite_key.split('==>')[0], 'cited':cite_key.split('==>')[1]}
print dt.prepContexts(d, t, p.titles, c).toxml()
