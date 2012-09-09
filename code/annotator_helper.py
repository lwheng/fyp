class annotator_helper:
  def printer(self, citing, cited, citStr, wholeText, citedLines):
    display = "<div style='width:100%'>"
    display += "<h2>" + citing + " cites " + cited + "</h2>"
    display += "<a href='http://aclweb.org/anthology/" + citing[0] + "/" + citing[0:3] + "/" + citing + ".pdf'>citing</a>"
    display += "<p>"
    display += "<a href='http://aclweb.org/anthology/" + cited[0] + "/" + cited[0:3] + "/" + cited+ ".pdf'>cited</a>"
    display += "<h2>citStr = " + citStr + "</h2>"
    display += "<div style='float:left; display:inline-block; width:40%; overflow:auto' class='div1'>"
    display += wholeText
    display += "</div>"
    display += "<div style='float:right; display:inline-block; width:50%; height:400px; overflow:auto' class='div2'>"
    for i in range(len(citedLines)):
      l = citedLines[i]
      display += "<div>" + str(i+1) + ". " + l + "</div>"
    display += "</div>"
    display += "</div>"
    return display
