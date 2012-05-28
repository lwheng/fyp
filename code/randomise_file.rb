#!/usr/bin/ruby
# Prints to stdout
abort "usage: randomise_file.rb <file>" if ARGV.size != 1

# get the lines:
lines = IO.readlines(ARGV[0])

# pick a random line, remove it, and print it
lines.size.times do 
  print lines.delete_at(rand(lines.size)).gsub("\s","").gsub("\n","") + ",\n"
end
