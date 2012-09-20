#!/bin/bash

INPUTFILE=$1
output=""

cat $INPUTFILE | while read LINE
do
  temp1=${LINE/,/}
  IFS='==>'
  for i in $temp1
  do
    if [ "$i" != "" ]
    then
      temp=$output
      output="$temp $i.txt "
      echo $output
    fi
  done
done
