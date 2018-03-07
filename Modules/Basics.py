from __future__ import division
import numpy as np
import pandas
import math
import os
import types
import h5py
from six.moves import cPickle as pickle

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

from ML_Tools.Plotting_And_Evaluation.Plotters import *
from ML_Tools.General.Misc_Functions import *

dirLoc = "../Data/"

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print 'radicand is negative. Exiting'
        exit()
    else:
        return math.sqrt(radicand)


def amsScan(inData, res=0.0001):
    best = [0,-1]
    for i in np.linspace(0.,1.,1./res):
        ams = AMS(np.sum(inData.loc[(inData['pred_class'] >= i) & sigVal, 'gen_weight']),
                  np.sum(inData.loc[(inData['pred_class'] >= i) & bkgVal, 'gen_weight']))
        if ams > best[1]:
            best = [i, ams]
    print best

def scoreTest(ensemble, weights, features, cut, name):
	testData = pandas.read_csv('../Data/test.csv')
	with open(dirLoc + 'inputPipe.pkl', 'r') as fin:
    	inputPipe = pickle.load(fin)

	testaData['pred_class'] = ensemblePredict(inputPipe.transform(testData[features].values.astype('float32')), ensemble, weights)    	

	testData['Class'] = 'b'
	testData.loc[testData.pred_class >= cut, 'Class'] = 's'

	testData.sort_values(by=['pred_class'], inplace=True)
	testData['RankOrder']=range(1, len(testData)+1)
	testData.sort_values(by=['EventId'], inplace=True)

	testData.to_csv(dirLoc + name + '_test.csv', columns=['EventId', 'RankOrder', 'Class'], index=False)