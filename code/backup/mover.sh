paperList=$1
source=$2
target=$3

cat $paperList | while read LINE
do
  cp $(find $source | grep $LINE) $target
done
