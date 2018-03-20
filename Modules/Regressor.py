from __future__ import division

from keras.models import Sequential,model_from_json, load_model

from ML_Tools.General.Ensemble_Functions import ensemblePredict, loadModel
from ML_Tools.General.Batch_Train import getFeature, batchEnsemblePredict

from six.moves import cPickle as pickle
import glob
import types
import numpy as np

class Regressor():
    def __init__(self):
        self.models = []
        self.weights = []
        self.size = 0
        self.inputPipe = None
        
    def _loadModel(self, cycle, location='train_weights/train_'): #Function to load a specified classifier
        cycle = int(cycle)
        model = load_model(location + str(cycle) + '.h5')
        return model
    
    def _getWeights(self, value, mode='rec'): #How the weight is calculated.
        if mode == 'rec':
            return 1/value #Reciprocal of metric is a simple way of assigning larger weights to better metrics
        elif mode == 'uni':
            return 1 #Uniform weighting
        else:
            print("Weight mode not recognised/supported")
            return 0

    def buildEnsemble(self, results, size, weighting='rec', verbose=True):
        self.models = []
        weights = []

        dtype = [('cycle', int), ('result', float)]
        values = np.sort(np.array([(i, result['loss']) for i, result in enumerate(results)], 
                                  dtype=dtype), order=['result'])

        for i in range(min([size, len(results)])):
            self.models.append(self._loadModel(values[i]['cycle']))
            weights.append(self._getWeights(values[i]['result'], mode=weighting))
            if verbose: print("Model {} is {} with loss = {}". format(i, values[i]['cycle'], values[i]['result']))

        weights = np.array(weights)
        self.weights = weights/weights.sum() #normalise weights
        self.size = len(self.models)
    
    def addInputPipe(self, pipe):
        self.inputPipe = pipe

    def predict(self, X, preProcX, n=-1): #Loop though each classifier and predict data class
        if not isinstance(self.inputPipe, types.NoneType): #Preprocess if necessary
            preProcX = self.inputPipe.transform(preProcX)
            X = np.append(X, preProcX, axis=1)
        
        pred = np.zeros((len(X), 1))
        
        if n == -1:
            n = len(self.models)+1
            
        models = self.models[0:n] #Use only specified number of classifiers
        weights = self.weights[0:n]/self.weights[0:n].sum() #Renormalise weights

        for i, model in enumerate(models): #Weight each model prediction
            pred += weights[i]*model.predict(X, verbose=0)
        
        return pred
    
    def save(self, filename, overwrite=False):
        if (len(glob.glob(filename + "*.json")) or len(glob.glob(filename + "*.h5")) or len(glob.glob(filename + "*.pkl"))) and not overwrite:
            print("Ensemble already exists with that name, call with overwrite=True to force save")
        
        else:
            os.system("rm " + filename + "*.json")
            os.system("rm " + filename + "*.h5")
            os.system("rm " + filename + "*.pkl")
            
            for i, model in enumerate(self.models):
                model.save(filename + '_' + str(i) + '.h5')
                
            with open(filename + '_weights.pkl', 'w') as fout:
                pickle.dump(self.weights, fout)
                
            if not isinstance(self.inputPipe, types.NoneType):
                with open(filename + '_inputPipe.pkl', 'w') as fout:
                    pickle.dump(inputPipe, fout)
                    
    def load(self, filename):
                
        self.models = []
        
        for model in glob.glob(filename + '_*.h5'):
            self.models.append(load_model(model))
            
        print len(self.models), "models loaded"
        
        with open(filename + '_weights.pkl', 'r') as fin:
            self.weights = weights = pickle.load(fin)
            
        try:
            print "Loading inputpipe"
            with open(filename + '_inputPipe.pkl', 'r') as fin:
                self.inputPipe = pickle.load(fin)
        except IOError:
            print "No inputpipe found"
            self.inputPipe = None