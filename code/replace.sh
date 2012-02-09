FILENAME=$1
cat $FILENAME | while read LINE
do
    echo $(echo $LINE | sed -e "s/\ ==>\ /==>/g") >> testing.txt
done
