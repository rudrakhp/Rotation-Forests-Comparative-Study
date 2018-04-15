#!/bin/bash
chmod +x generate.sh
rm ./compare.txt
for i in {1..35}
do
    for j in {1..15}
    do
        python RFComparativeStudy.py $j $i
    done
done