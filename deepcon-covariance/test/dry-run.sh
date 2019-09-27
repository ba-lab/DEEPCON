#!/bin/bash

set -e
rm -f ./*rr

echo "Predicting.."
python ../deepcon-covariance.py --aln ./16pkA0.aln --rr ./16pkA0.rr

echo "Evaluating.."
./coneva-lite.pl -pdb ./16pkA.pdb -rr ./16pkA0.rr

echo "Predicting.."
python ../deepcon-covariance.py --aln ./1a0tP0.aln --rr ./1a0tP0.rr

echo "Evaluating.."
./coneva-lite.pl -pdb ./1a0tP.pdb -rr ./1a0tP0.rr
