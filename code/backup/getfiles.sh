FILENAME=$1
LIST=$2

cat $FILENAME | while read LINE
do
    echo $(echo $LINE | cut -d '>' -f2) >> THISISATEMPFILE.txt
done

cat THISISATEMPFILE.txt | while read LINE1
do
    cat $(cat $LIST | grep $LINE1) > $LINE1.txt
    open -a "Sublime Text 2" $LINE1.txt
done
rm THISISATEMPFILE.txt