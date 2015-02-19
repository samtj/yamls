from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as pyplot
from facial_keypoint import helper
import numpy as np
import cPickle as pickle
import os.path

def initialize_conv():

    # if train's already done
    if os.path.exists('net2.pickle'):
        with open(r"net2.pickle", "rb") as input_file:
            net2 = pickle.load(input_file)
        return net2
    
    # use the cuda-convnet implementations of conv and max-pool layer
    #Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
    #MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer
    
    # workaround to not use GPU
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
    
    net2 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', Conv2DLayer),
            ('pool1', MaxPool2DLayer),
            ('conv2', Conv2DLayer),
            ('pool2', MaxPool2DLayer),
            ('conv3', Conv2DLayer),
            ('pool3', MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, 96, 96),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
        hidden4_num_units=500, hidden5_num_units=500,
        output_num_units=30, output_nonlinearity=None,
    
        update_learning_rate=0.01,
        update_momentum=0.9,
    
        regression=True,
        max_epochs=1000, # 1000 will take a while
        verbose=1,
        )
    
    X, y = helper.load2d()  # load 2-d data
    net2.fit(X, y)
    
    with open('net2.pickle', 'wb') as f:
        pickle.dump(net2, f, -1)
    return net2

def main():
    net2 = initialize_conv()
    
    # Training for 1000 epochs will take a while.  We'll pickle the
    # trained model so that we can load it back later:
    #with open('net2.pickle', 'wb') as f:
    #    pickle.dump(net2, f, -1)
        
    #with open(r"someobject.pickle", "rb") as input_file:
    #    e = cPickle.load(input_file)
    # Training this neural net is much more computationally costly than the first one we trained.
    # It takes around 15x as long to train; those 1000 epochs take more than 20 minutes on even a powerful GPU.
    

if __name__ == "__main__":
    main()
