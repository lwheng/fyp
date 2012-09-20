# This script takes in the listOfFiles, and generate all the files
# from the corpus by concat-ing them together

LISTOFFILE=$1
FINDLIST=$2
cat $LISTOFFILE | while read LINE1
do
    cat $(cat $FINDLIST | grep $LINE1) > ~/Desktop/Files/$LINE1.txt
done