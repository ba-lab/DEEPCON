### DEEPCON using Covariance features as input

Trained and validated using the 3456 proteins in the DeepCov dataset with the covariance features (441 channels) as input.

#### Predict
```bash
python ../deepcon-covariance.py --aln ./16pkA0.aln --rr ./16pkA0.rr
```

#### Evaluate
```bash
./coneva-lite.pl -pdb ./16pkA.pdb -rr ./16pkA0.rr
```
