paperList=$1

cat $paperList | while read LINE
do
  find /Users/lwheng/Downloads/fyp/antho | grep $LINE
done
