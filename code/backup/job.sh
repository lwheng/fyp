#!/bin/bash

# Input: Interlink file
# Output: Runs jobs and computes the resuits
# Usage: ./job.sh <interlink file> <no of lines to run>

interlink=$1
lines=$2
head -n $2 $interlink | while read LINE
do
  echo "Computing for $LINE"
  citing=/Users/lwheng/Downloads/fyp/txt/$(echo $LINE | tr " " "\n" | head -n 1).txt
  cited=/Users/lwheng/Downloads/fyp/txt/$(echo $LINE | tr " " "\n" | tail -n 1).txt
  python sim.py -w -n 20 -1 $citing -2 $cited
  echo
done
