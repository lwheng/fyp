require 'open-uri'
require 'rexml/document'
require 'text'
require 'getopt/std'
include REXML
include Text

$annotationMasterFile = ""

def grabContext()
  openfile = File.open($annotationMasterFile,"r")
  while (line = openfile.gets)
    cite_key = line[0..18].strip
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]
    
    # cited paper
    citedFile = "http://wing.comp.nus.edu.sg/~antho/" + cited[0] + "/" + cited[0,3] + "/" + cited 
    titleCited = ""
    begin
      file = open("#{citedFile}.bib","r")
      regexTitle = /(^title)\s*=\s*\{(.*)\}/
      while (l = file.gets)
        input = l.strip()
        if match = regexTitle.match(input)
          titleCited = match[2].strip
        end
      end
      file.close()
    rescue => error
      # No bib file
      begin
        file = open("#{citedFile}-pdfbox-seersuite.txt","r")
        data = file.read
        file.close()
        root = (Document.new data).root
        content = root.elements
        if content["title"]
          titleCited = content["title"].text.strip
        end
      rescue => error1
        # No seersuite file either
        # try -final.xml
        begin
          file = open("#{citedFile}-final.xml","r")
          data = file.read
          file.close()
          root = (Document.new data).root
          title = root.elements["teiHeader"].elements["fileDesc"].elements["titleStmt"].elements["title"]
          titleCited = title.text.strip
        rescue => error2
          begin
            # try parscit-section.xml
            file = open("#{citedFile}-parscit-section.xml","r")
            data = file.read
            file.close()
            root = (Document.new data).root
            variant = root.elements['algorithm'].elements['variant']
            if variant.elements['title']
              title = variant.elements['title']
            elsif variant.elements['note']
              title = variant.elements['note']
            end
            titleCited = title.text.gsub("\n"," ").strip
          rescue => error3
            # puts "NO TITLE for #{citedFile}!!"
            puts "#{cite_key}!=<NOFILE>"
            next
          end
        end
      end
    end
    
    # citing paper
    citationCiting = nil
    citingFile = "http://wing.comp.nus.edu.sg/~antho/" + citing[0] + "/" + citing[0,3] + "/" + citing 
    distance = 314159265358979323846
    begin
      file = open("#{citingFile}-parscit.xml")
      data = file.read
      file.close()
      root = (Document.new data).root
      citationList = root.elements["algorithm"].elements["citationList"]
      citations = citationList.elements
      citations.each do |v|
        if v.attributes['valid'].eql?("true")
          if v.elements['title']
            title = v.elements['title']
          elsif v.elements['booktitle']
            title = v.elements['booktitle']
          elsif v.elements['journal']
            title = v.elements['journal']
          elsif v.elements['note']
            title = v.elements['note']
          elsif v.elements['institution']
            title = v.elements['institution']
          elsif v.elements['publisher']
            title = v.elements['publisher']
          elsif v.elements['tech']
            title = v.elements['tech']
          elsif v.elements['marker']
            title = v.elements['marker']
          else
            title = nil
          end
          if title
            check = Levenshtein.distance(titleCited, title.text)
            if check < distance
              citationCiting = v
              distance = check
            end
          else
            # cannot get a title
          end
        else
          # citation not valid. skip?
        end
      end
      # print output
      if citationCiting
        if citationCiting.elements['contexts']
          if citationCiting.elements['contexts'].elements.size > 0
            contexts = citationCiting.elements['contexts'].elements
            contexts.each do |v|
              puts "#{cite_key}!=#{v}"
            end
          else
            puts "#{cite_key}!=<NOCONTEXTTAG>"
          end
        else
          puts "#{cite_key}!=<NOCONTEXTSTAG>"
        end
      else
        puts "#{cite_key}!=<NOCITATIONCITING>"
      end
    rescue => error3
      # no parscit-section.xml
      puts "#{cite_key}!=<NOPARSCITSECTION>"
      next
    end
    # exit()
  end
end

def usage()
  puts "USAGE: ruby #{$0} -f <annotationMasterFile>"
end

def main()
  opt = Getopt::Std.getopts("f:")

  if opt["f"]
    $annotationMasterFile = opt['f']
  else
    usage
    exit()
  end

  grabContext
end

main
