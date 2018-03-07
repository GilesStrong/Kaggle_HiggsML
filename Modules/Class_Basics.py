from __future__ import division

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import Sequential

from six.moves import cPickle as pickle
import timeit
import types
import numpy as np
import pandas

from rep.estimators import XGBoostClassifier
from xgboost import XGBClassifier

from ML_Tools.General.PreProc import *
from ML_Tools.General.Ensemble_Functions import *
from ML_Tools.Plotting_And_Evaluation.Bootstrap import mpRun
from ML_Tools.General.Training import *
from ML_Tools.General.Batch_Train import *

from Class_Features import *

def getClassifier(version=None, nIn=None, compileArgs=None, nOut=1):
    model = Sequential()
    depth = None
    width = None

    if version == "modelRelu":
        depth = 3
        width = 100
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal'))
            model.add(Activation('relu'))

    elif version == "modelSelu":
        depth = 3
        width = 100
        model.add(Dense(width, input_dim=nIn, kernel_initializer='VarianceScaling'))
        model.add(Activation('selu'))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='VarianceScaling'))
            model.add(Activation('selu'))

    elif version == "modelSwish":
        depth = 3
        width = 100
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal'))
        model.add(Activation('swish'))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal'))
            model.add(Activation('swish'))
            
    if nOut == 1:
        model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
    else:
        model.add(Dense(nOut, activation='softmax', kernel_initializer='glorot_normal'))

    if 'lr' not in compileArgs: compileArgs['lr'] = 0.001
    if compileArgs['optimizer'] == 'adam':
        optimiser = Adam(lr=compileArgs['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
    model.compile(loss=compileArgs['loss'], optimizer=optimiser)
    return model