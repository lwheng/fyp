require 'rexml/document'
require 'text'
require 'getopt/std'
include REXML
include Text

$annotationMasterFile = ""

def annotations500()
  openfile = File.open($annotationMasterFile,"r")
  numOfCites = 500
  while (line = openfile.gets) && numOfCites > 0
    cite_key = line[0..18].strip
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]

    # Get title of cited paper
    begin
      citedFile = "/Users/lwheng/Downloads/fyp/antho/#{cited[0]}/#{cited[0,3]}/#{cited}"
      if File.exist?("#{citedFile}.bib")
        file = open("#{citedFile}.bib", "r")
        regexTitle = /(^title)\s*=\s*\{(.*)\}/
        while (l=file.gets)
          input = l.strip()
          if match = regexTitle.match(input)
            titleCited = match[2].strip
          end
        end
      elsif File.exist?("#{citedFile}-parscit-section.xml")
        file = open("#{citedFile}-parscit-section.xml","r")
        data = file.read
        file.close()
        root = (Document.new data).root
        variant = root.elements['algorithm'].elements['variant']
        if variant.elements['title']
          title = variant.elements['title']
        elsif variant.elements['note']
          title = variant.elements['note']
        else
          title = nil
        end
        titleCited = title.text.gsub("\n"," ").strip
      elsif File.exist?("#{citedFile}-final.xml")
        file = open("#{citedFile}-final.xml","r")
        data = file.read
        file.close()
        root = (Document.new data).root
        titleStmt = root.elements['teiHeader'].elements['fileDesc'].elements['titleStmt']
        if titleStmt.elements['title']
          title = titleStmt.elements['title']
        else
          title = nil
        end
        titleCited = title.text.gsub("\n"," ").strip
      else
        titleCited = ""
        next
      end
      
      if titleCited.size < 10 || titleCited.size > 120
        next
      end
      
      # Check citing for parscit file
      citationCiting = nil
      citingFile = "/Users/lwheng/Downloads/fyp/antho/#{citing[0]}/#{citing[0,3]}/#{citing}"
      distance = 314159265358979323846
      if File.exist?("#{citingFile}-parscit.xml")
        file = open("#{citingFile}-parscit.xml","r")
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
        
        if citationCiting
          if citationCiting.elements['contexts']
            if citationCiting.elements['contexts'].elements.size > 0
              # puts "#{cite_key} #{500-numOfCites}. #{titleCited}"
              puts cite_key
              numOfCites = numOfCites - 1
            else
              next
            end
          else
            next
          end
        else
          next
        end
      else
        next
      end
    rescue => error2
      next
      # exit()
    end
    # exit()
  end
end

def usage()
  puts "ruby #{$0} -f <annotationMasterFile>"
end

def main()
  opt = Getopt::Std.getopts("f:")

  if opt['f']
    $annotationMasterFile = opt['f']
  else
    usage
    exit()
  end
  annotations500
end

main
