#!/usr/bin/python
# Badri Adhikari, 4-2-2019
# https://github.com/badriadhikari/

################################################################################
from keras.layers import *
from keras.models import Model
import os, sys, datetime
import tensorflow as tf
import numpy as np
K.set_image_data_format('channels_last')
import argparse

################################################################################
n_channels = 441
pathname = os.path.dirname(sys.argv[0])
model_weights_file_name = pathname + '/weights-rdd-covariance.hdf5'

################################################################################
def aanum(ch):
    aacvs = [999, 0, 3, 4, 3, 6, 13, 7, 8, 9, 21, 11, 10, 12, 2,
            21, 14, 5, 1, 15, 16, 21, 19, 17, 21, 18, 6]
    if ch.isalpha():
        return aacvs[ord(ch) & 31]
    return 20

################################################################################
# Python reimplementation of the original code by David Jones @ UCL
def cov21stats(file_aln):
    alnfile = np.loadtxt(file_aln, dtype = str)
    nseq = len(alnfile)
    L = len(alnfile[0])
    aln = np.zeros(((nseq, L)), dtype = int)
    for i, seq in enumerate(alnfile):
        for j in range(L):
            aln[i, j] = aanum(seq[j])
     # Calculate sequence weights
    idthresh = 0.38
    weight = np.ones(nseq)
    for i in range(nseq):
        for j in range(i+1, nseq):
            nthresh = int(idthresh * L)
            for k in range(L):
                if nthresh > 0:
                    if aln[i, k] != aln[j, k]:
                        nthresh = nthresh - 1
            if nthresh > 0:
                weight[i] += 1
                weight[j] += 1
    weight = 1/weight
    wtsum  = np.sum(weight)
    # Calculate singlet frequencies with pseudocount = 1
    pa = np.ones((L, 21))
    for i in range(L):
        for a in range(21):
            pa[i, a] = 1.0
        for k in range(nseq):
            a = aln[k, i]
            if a < 21:
                pa[i, a] = pa[i, a] + weight[k]
        for a in range(21):
            pa[i, a] = pa[i, a] / (21.0 + wtsum)
    # Calculate pair frequencies with pseudocount = 1
    pab = np.zeros((L, L, 21, 21))
    for i in range(L):
        for j in range(L):
            for a in range(21):
                for b in range(21):
                    pab[i, j, a, b] = 1.0 / 21.0
            for k in range(nseq):
                a = aln[k, i]
                b = aln[k, j]
                if (a < 21 and b < 21):
                    pab[i, j, a, b] = pab[i, j, a, b] + weight[k];
            for a in range(21):
                for b in range(21):
                    pab[i, j, a, b] = pab[i, j, a, b] / (21.0 + wtsum)
    final = np.zeros((L, L, 21, 21))
    for a in range(21):
        for b in range(21):
            for i in range(L):
                for j in range(L):
                    final[i, j, a, b] = pab[i, j, a, b] - pa[i, a] * pa[j, b]
    final4d = final.reshape(1, L, L, 21 * 21)
    return final4d

################################################################################
def main(aln, file_rr):
    print("Start " + str(sys.argv[0]) + " - " + str(datetime.datetime.now()))
    global n_channels
    global model_weights_file_name
    if not os.path.isfile(model_weights_file_name):
        print('Model weights file ' + model_weights_file_name + ' is absent!\n')
        print('Please download from https://github.com/badriadhikari/DEEPCON/')
        sys.exit(1)
    sequence = ''
    print ('')
    print ('Read sequence[0] from aln..')
    with open(aln) as f:
        sequence = f.readline()
        sequence = sequence.strip()
    L = len(sequence)
    if L < 20:
        print ("ERROR!! Too short sequence!!")
    print ('')
    print ('Convert aln to covariance matrix.. patience..')
    sys.stdout.flush()
    X = cov21stats(aln)
    if X.shape != (1, L, L, n_channels):
        print('Unexpected shape from cov21stats!')
        print(X.shape)
        sys.exit(1)
    print ('')
    print ('Build a model of the size of the input (and not bigger)..')
    sys.stdout.flush()
    dropout_value = 0.3
    input_original = Input(shape = (L, L, n_channels))
    tower = input_original
    # Start - Maxout Layer for DeepCov dataset
    input = BatchNormalization()(input_original)
    input = Activation('relu')(input)
    input = Convolution2D(128, 1, padding = 'same')(input)
    input = Reshape((L, L, 128, 1))(input)
    input = MaxPooling3D(pool_size = (1, 1, 2))(input)
    input = Reshape((L, L, 64 ))(input)
    tower = input
    # End - Maxout Layer
    n_channels = 64
    d_rate = 1
    for i in range(32):
        block = BatchNormalization()(tower)
        block = Activation('relu')(block)
        block = Convolution2D(64, 3, padding = 'same')(block)
        block = Dropout(dropout_value)(block)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, 3, dilation_rate=(d_rate, d_rate), padding = 'same')(block)
        tower = add([block, tower])
        if d_rate == 1:
            d_rate = 2
        elif d_rate == 2:
            d_rate = 4
        else:
            d_rate = 1
    tower = BatchNormalization()(tower)
    tower = Activation('relu')(tower)
    tower = Convolution2D(1, 3, padding = 'same')(tower)
    output = Activation('sigmoid')(tower)
    model = Model(input_original, output)
    print ('')
    print ('Load weights from ' + model_weights_file_name + '..')
    model.load_weights(model_weights_file_name)
    print ('')
    print ('Predict..')
    sys.stdout.flush()
    P1 = model.predict(X)
    P2 = P1[0, 0:L, 0:L]
    P3 = np.zeros((L, L))
    for p in range(0, L):
        for q in range(0, L):
            P3[q, p] = (P2[q, p] + P2[p, q]) / 2.0
    print ('')
    print 'Write RR file ' + file_rr + '.. '
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    for i in range(0, L):
        for j in range(i, L):
            if abs(i - j) < 5:
                continue
            rr.write("%i %i 0 8 %.5f\n" %(i+1, j+1, P3[i][j]))
    rr.close()
    print("Done " + str(sys.argv[0]) + " - " + str(datetime.datetime.now()))

################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aln'
        , help = 'Input Alignment file'
        , required = True
    )
    parser.add_argument('--rr'
        , help = 'Output RR file (CASP format)'
        , required = True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    main(arguments['aln'], arguments['rr'])
