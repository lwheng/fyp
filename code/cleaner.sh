# This is script is a helper script, to clean all files of non-ascii characters.

# Usage:
# Run this:  ./cleaner.sh INPUT OUTPUT
# INPUT: File directory of the dirty files

INPUT=$1
find $INPUT > ThisIsATempFileYouCannotMiss
cat ThisIsATempFileYouCannotMiss | while read FILENAME
do
	echo "STARTED on $FILENAME"
	python cleaner.py $FILENAME $FILENAME.cleaned
	echo "END"
done

rm ThisIsATempFileYouCannotMiss

# python cleaner.py $INPUT ThisIsATempFileYouCannotMiss
# sort ThisIsATempFileYouCannotMiss > $OUTPUT
# rm ThisIsATempFileYouCannotMiss