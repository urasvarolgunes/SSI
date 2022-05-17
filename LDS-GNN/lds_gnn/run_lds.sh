for dataset in fin_small_google fin_small_glove fin_small_fast \
fin_large_google fin_large_glove fin_large_fast \
applestore_google applestore_glove

do

echo DATASET: $dataset
python lds.py -d=$dataset
echo

done