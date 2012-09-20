FILENAME=$1
LIST=$2
cat $(cat $LIST | grep $FILENAME) > $FILENAME.txt
open -a "Sublime Text 2" $FILENAME.txt
