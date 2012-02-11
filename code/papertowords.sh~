# This script will take in the interlink file as input
# and output a file with the list of paper file names
# E.g.
# 
# A00-1001 ==> J82-3002
# A00-1002 ==> C90-3057
# A00-1002 ==> P98-1080
# A00-1003 ==> P98-1066
# A00-1003 ==> P99-1027
# 
# becomes --->
# 
# A001-1001
# A001-1002
# A001-1003
# A001-1004

INPUT=$1
OUTPUT=$2
python cleaner.py $INPUT ThisIsATempFileYouCannotMiss
sort ThisIsATempFileYouCannotMiss > $OUTPUT
rm ThisIsATempFileYouCannotMiss