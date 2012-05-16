# INPUT: Directory path to scientic papers in text
# PROGRESS: Runs WING's Parcit to extract all citations
# OUTPUT: Scientific papers in XML format, citing sentence easily accessible

INDIR=$1
OUTDIR=$2
ls $INDIR | while read LINE
do
INFILE=$DIR$LINE
TEMP=${LINE/\.txt/\.xml}
OUTFILE=$OUTDIR$TEMP
/home/wing.nus/services/parscit/tools/bin/citeExtract.pl -m extract_all $INFILE $OUTFILE
done
