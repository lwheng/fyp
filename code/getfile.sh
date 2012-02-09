FILENAME=$1
cat $(cat list | grep $FILENAME) > $FILENAME.txt
open -a "Sublime Text 2" $FILENAME.txt
