from pylearn2.train import Train
from pylearn2.models import mlp, maxout
#from pylearn2.space import Conv2DSpace
from pylearn2.termination_criteria import MonitorBased, EpochCounter
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.costs.mlp.dropout import Dropout as DropoutCost
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor
#from theano.sandbox.cuda.type import CudaNdarrayType
#from theano.misc.pycuda_utils import to_gpuarray, to_cudandarray

import warnings
import numpy as np
import theano
import theano.tensor
from pylearn2.datasets.sparse_dataset import SparseDataset

import logging

class ClassMap:
    """
    A tool for converting classification datasets having
    {-1,1} (or any other kind) as its set of categories.
    It also contains a reverse mapping for converting back to the
    original class set at prediction time. 
    """
    def __init__(self, y):
        classes = np.unique(y)
        self.min = np.min(classes)
        self._map = np.zeros( np.max(classes)-self.min+1, np.int ) 
        self._invmap = np.zeros(len(classes), np.int)
        for i, c in  enumerate(classes):
            self._map[c-self.min] = i
            self._invmap[i] = c
        
    def map(self, y ): 
        return self._map[y-self.min]
    
    def invmap(self, y ): 
        return self._invmap[y]



class Classifier:
        
    def _make_dataset(self,x,y):
        y = np.asarray(y, dtype=np.int)
        if not hasattr( self, "classmap" ):
            self.classmap = ClassMap(y)
            
        ds = DenseDesignMatrix(X=x, y=self.classmap.map(y) )
        ds.convert_to_one_hot()
        return ds
    
    def fit(self, x, y=None ):
        ds = self._make_dataset(x.todense(), y)
        return self.train( ds )

    def train(self, dataset):
        self._build_model(dataset)
        
        x_variable = theano.tensor.matrix()
        y = self.model.fprop(x_variable)
        self.fprop = theano.function([x_variable], y)
        
        momentum_adjustor = MomentumAdjustor(
            start= 1,
            saturate= 250,
            final_momentum= .7
        )
        
        train = Train( dataset, self.model, 
            self.algorithm, extensions=[momentum_adjustor] )
        logging.getLogger("pylearn2").setLevel(logging.WARNING)
        train.main_loop()
        logging.getLogger("pylearn2").setLevel(logging.INFO)
    
        return self
    
    def _build_model(self, dataset):
        raise NotImplemented('Please, override this function.')
    
    def predict_proba(self, x):
        return self.fprop(x.data)
        
    def predict(self, x ):
        y_prob = self.predict_proba(x.data)
        idx = np.argmax(y_prob,1)
        y = self.classmap.invmap(idx)
        return y

    def set_valid_info(self,x,y):
        """
        Optional. But, if there is no valid_dataset, training will
        stop after a fixed number of iterations. 
        If you already have a valid pylearn2 dataset, you can directly 
        assign the valid_dataset attribute.
        """
        self.valid_dataset = self._make_dataset(x, y)
        
class MaxoutClassifier(Classifier):
    
    def __init__(self,
         
        num_units = (100,100),
        num_pieces = 2,
        learning_rate = 0.1, 
        irange = 0.005,
        W_lr_scale = 1.,
        b_lr_scale = 1., 
        max_col_norm = 1.9365):
        
        self.__dict__.update( locals() )
        del self.self
    
    def _broadcast_param(self, param_name, layer ):
        """
        helper function to distinguish between fixed parameter
        or a different parameter for each layer
        """
        param = getattr(self, param_name, None )
        try:
            assert len(self.num_units) == len(param), '%s must have the same length as num_units or be a scalar.'%param_name
            return param[layer]
        except TypeError: # should be raised by len(param) if it is not a list
            return param
    
    def _build_model(self, dataset):
        # is there a more standard way to get this information ?
        n_features = dataset.X.shape[1] 
        #n_classes = len(np.unique( dataset.y ) )
        n_classes = 5
        
        layers = []
        for i, num_units in enumerate(self.num_units):
            
            layers.append( maxout.Maxout (
                layer_name= 'h%d'%i,
                num_units= num_units,
                num_pieces= self._broadcast_param('num_pieces', i),
                W_lr_scale= self._broadcast_param('W_lr_scale', i),
                b_lr_scale= self._broadcast_param('b_lr_scale', i),
                irange    = self._broadcast_param('irange', i),
                max_col_norm= self.max_col_norm,
            ))
            
            print "layer %d"%i
            for key, val in layers[-1].__dict__.iteritems():
                print key, val
            print
                
#            print 'layer %d'%i
#            for key, val in layers[-1].__dict__.items():
#                print key, val
#            print
            
        layers.append(  mlp.Softmax (
            max_col_norm= self.max_col_norm,
            layer_name= 'y',
            n_classes= n_classes,
            irange= self.irange,
        ))
        
                
        self.model = mlp.MLP(
            batch_size = 100,
            layers = layers,
            nvis=n_features,
        )
        
        
        try:
            monitoring_dataset= {'valid' : self.valid_dataset}
            monitor = MonitorBased( channel_name= "valid_y_misclass", prop_decrease= 0., N= 100)
            print 'using the valid dataset'
        except AttributeError: 
            warnings.warn('No valid_dataset. Will optimize for 1000 epochs')
            monitoring_dataset = None
            monitor = EpochCounter(1000)
        
        #added initial momentum         
        init_momentum= .5
        momentum_rule = learning_rule.Momentum(init_momentum)

        self.algorithm = sgd.SGD(
            learning_rate= self.learning_rate,
            learning_rule= momentum_rule,
            train_iteration_mode = 'even_sequential',

            monitoring_dataset= monitoring_dataset,
            cost= DropoutCost(
                input_include_probs= { 'h0' : .8 },
                input_scales= { 'h0': 1. }
            ),
            termination_criterion= monitor,
            update_callbacks= sgd.ExponentialDecay(
                decay_factor= 1.00004,
                min_lr= .000001
            ),
        )